"""Locate, read, and content-address the packaged bootstrap script.

The script is resolved through :mod:`importlib.resources` rather than by walking up from
``__file__`` so it is found when ``hpcctl`` is installed as a wheel, not only from a source
checkout.

The S3 key is **content-addressed** -- it embeds the first 8 hex characters of the script's
SHA-256. With a stable key, editing the script and re-booting a cluster can leave nodes fetching a
cached or half-replaced object, producing an inconsistent fleet with no error anywhere. A digest
in the key makes "which script did this cluster actually run" answerable from the config alone,
and makes re-upload idempotent: same bytes, same key, no-op.
"""

import hashlib
from importlib.resources import as_file, files
from pathlib import Path

from hpcctl.config import Settings

SCRIPT_NAME: str = "install_engine_deps.sh"
"""Filename of the packaged bootstrap script."""

DIGEST_PREFIX_LENGTH: int = 8
"""Number of hex characters of the SHA-256 embedded in the S3 key."""


def bootstrap_path() -> Path:
    """Return the filesystem path to the packaged bootstrap script.

    Returns:
        The path to ``install_engine_deps.sh`` inside the installed package.
    """
    resource = files("hpcctl").joinpath("bootstrap").joinpath(SCRIPT_NAME)
    with as_file(resource) as path:
        return Path(path)


def bootstrap_text() -> str:
    """Return the bootstrap script's contents.

    Returns:
        The exact script text, read as UTF-8.
    """
    resource = files("hpcctl").joinpath("bootstrap").joinpath(SCRIPT_NAME)
    return resource.read_text(encoding="utf-8")


def bootstrap_digest() -> str:
    """Return the SHA-256 hex digest of the bootstrap script.

    Returns:
        The full 64-character lowercase hex digest.
    """
    return hashlib.sha256(bootstrap_text().encode("utf-8")).hexdigest()


def bootstrap_key(settings: Settings) -> str:
    """Return the content-addressed S3 key for the bootstrap script.

    Args:
        settings: Resolved settings supplying the key prefix.

    Returns:
        A key such as ``hpcctl/bootstrap/install_engine_deps-1a2b3c4d.sh``.
    """
    short = bootstrap_digest()[:DIGEST_PREFIX_LENGTH]
    stem = SCRIPT_NAME.removesuffix(".sh")
    prefix = settings.bootstrap_prefix.strip("/")
    return f"{prefix}/{stem}-{short}.sh"


def bootstrap_s3_url(settings: Settings) -> str:
    """Return the content-addressed S3 URL for the bootstrap script.

    Args:
        settings: Resolved settings supplying the bucket and key prefix.

    Returns:
        An ``s3://`` URL suitable for ``CustomActions.OnNodeConfigured.Script``.
    """
    return f"s3://{settings.bootstrap_bucket}/{bootstrap_key(settings)}"


def upload_argv(settings: Settings) -> list[str]:
    """Build the ``aws s3 cp`` command that publishes the bootstrap script.

    Args:
        settings: Resolved settings supplying the destination.

    Returns:
        The ``aws`` argument vector.
    """
    return [
        "aws",
        "s3",
        "cp",
        str(bootstrap_path()),
        bootstrap_s3_url(settings),
        "--region",
        settings.region,
    ]
