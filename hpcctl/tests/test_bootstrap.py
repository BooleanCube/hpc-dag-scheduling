"""Tests for the packaged bootstrap script and its content addressing.

The apt-hygiene assertions are regexes on purpose: they are the rules most likely to erode under
a well-meaning edit, and each one prevents a specific hung-node failure rather than expressing a
style preference.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from hpcctl.config import Settings, load_settings
from hpcctl.generators.bootstrap import (
    DIGEST_PREFIX_LENGTH,
    bootstrap_digest,
    bootstrap_key,
    bootstrap_path,
    bootstrap_s3_url,
    bootstrap_text,
    upload_argv,
)


class TestPackaging:
    def test_path_resolves_via_importlib_resources(self) -> None:
        """Resolved as a package resource so it works from a wheel, not just a checkout."""
        path = bootstrap_path()
        assert path.is_file()
        assert path.name == "install_engine_deps.sh"

    def test_lives_inside_the_package(self) -> None:
        assert bootstrap_path().parent.name == "bootstrap"
        assert bootstrap_path().parent.parent.name == "hpcctl"

    def test_text_is_non_empty(self) -> None:
        assert len(bootstrap_text()) > 1000

    def test_text_matches_the_file_on_disk(self) -> None:
        assert bootstrap_text() == bootstrap_path().read_text(encoding="utf-8")

    def test_script_is_committable(self) -> None:
        """No .gitignore rule touches *.sh, which is why the script is not a template."""
        assert bootstrap_path().suffix == ".sh"


class TestSyntax:
    def test_passes_bash_syntax_check(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(bootstrap_path())], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
    def test_passes_shellcheck(self) -> None:
        """Optional: shellcheck is absent on the dev VM, so its absence never fails the suite."""
        result = subprocess.run(
            ["shellcheck", str(bootstrap_path())], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout

    def test_has_a_bash_shebang(self) -> None:
        assert bootstrap_text().startswith("#!/usr/bin/env bash\n")


class TestNonInteractiveHygiene:
    """Each assertion prevents a specific way node configuration hangs or dies."""

    def test_strict_mode(self) -> None:
        assert re.search(r"^set -euo pipefail$", bootstrap_text(), re.M)

    def test_debian_frontend_noninteractive(self) -> None:
        assert re.search(r"^export DEBIAN_FRONTEND=noninteractive$", bootstrap_text(), re.M)

    def test_force_confold(self) -> None:
        """DEBIAN_FRONTEND does not suppress dpkg's keep-or-replace prompt; this does."""
        assert "--force-confold" in bootstrap_text()

    def test_force_confdef(self) -> None:
        assert "--force-confdef" in bootstrap_text()

    def test_needrestart_is_suppressed(self) -> None:
        text = bootstrap_text()
        assert "NEEDRESTART_MODE=a" in text
        assert "NEEDRESTART_SUSPEND=1" in text

    def test_no_bare_apt_install(self) -> None:
        """Apt is a human-facing wrapper; apt-get is the stable scripting interface."""
        assert not re.search(r"(?<!-)\bapt install\b", bootstrap_text())

    def test_no_bare_apt_get_without_yes(self) -> None:
        text = bootstrap_text()
        for match in re.finditer(r"apt-get install([^\n]*)", text):
            tail = match.group(1)
            assert "APT_OPTS" in tail or "-y" in tail

    def test_no_upgrade(self) -> None:
        """Unbounded runtime and unrelated changes during node boot is what you least want."""
        assert not re.search(r"apt-get\s+(dist-)?upgrade", bootstrap_text())

    def test_update_is_retried(self) -> None:
        """One transient mirror failure under set -e would otherwise kill the node."""
        text = bootstrap_text()
        assert "apt_update_with_retry" in text
        assert re.search(r"for attempt in", text)

    def test_is_idempotent_via_a_version_stamped_marker(self) -> None:
        text = bootstrap_text()
        assert "BOOTSTRAP_VERSION=" in text
        assert "bootstrap.v${BOOTSTRAP_VERSION}.done" in text

    def test_marker_can_be_overridden(self) -> None:
        assert "--force" in bootstrap_text()

    def test_works_as_root_and_unprivileged(self) -> None:
        text = bootstrap_text()
        assert 'SUDO=""' in text
        assert '[ "$(id -u)" -ne 0 ]' in text

    def test_verifies_the_toolchain_at_the_end(self) -> None:
        """A node that cannot build the engine must fail at configure time, not at first job."""
        text = bootstrap_text()
        for tool in ("gcc", "cmake", "mpicc", "protoc"):
            assert tool in text
        assert "/usr/include/nlohmann/json.hpp" in text


class TestPackageList:
    @pytest.mark.parametrize(
        "package",
        [
            "build-essential",
            "cmake",
            "ninja-build",
            "git",
            "pkg-config",
            "ca-certificates",
            "curl",
            "unzip",
            "openmpi-bin",
            "libopenmpi-dev",
            "nlohmann-json3-dev",
            "libprotobuf-dev",
            "protobuf-compiler",
        ],
    )
    def test_required_package_is_present(self, package: str) -> None:
        assert re.search(rf"^\s+{re.escape(package)}$", bootstrap_text(), re.M)

    @pytest.mark.parametrize("package", ["gdb", "valgrind", "clang-format", "clang-tidy"])
    def test_dev_package_is_optional(self, package: str) -> None:
        assert package in bootstrap_text()

    def test_dev_tools_are_off_by_default(self) -> None:
        assert 'WITH_DEV_TOOLS="no"' in bootstrap_text()

    def test_dev_tools_are_opt_in(self) -> None:
        assert "--with-dev-tools" in bootstrap_text()


class TestContentAddressing:
    def test_digest_is_a_sha256_hex_string(self) -> None:
        digest = bootstrap_digest()
        assert len(digest) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", digest)

    def test_digest_is_stable_across_calls(self) -> None:
        assert bootstrap_digest() == bootstrap_digest()

    def test_key_embeds_the_digest_prefix(self, settings: Settings) -> None:
        short = bootstrap_digest()[:DIGEST_PREFIX_LENGTH]
        assert bootstrap_key(settings).endswith(f"install_engine_deps-{short}.sh")

    def test_key_uses_the_configured_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_BOOTSTRAP_PREFIX", "custom/path")
        assert bootstrap_key(load_settings(live=False)).startswith("custom/path/")

    def test_prefix_slashes_are_normalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_BOOTSTRAP_PREFIX", "/leading/and/trailing/")
        key = bootstrap_key(load_settings(live=False))
        assert not key.startswith("/")
        assert "//" not in key

    def test_s3_url_uses_the_bucket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_BOOTSTRAP_BUCKET", "my-bucket")
        url = bootstrap_s3_url(load_settings(live=False))
        assert url.startswith("s3://my-bucket/hpcctl/bootstrap/install_engine_deps-")

    def test_url_embeds_the_first_eight_digest_characters(self, settings: Settings) -> None:
        assert bootstrap_digest()[:8] in bootstrap_s3_url(settings)

    def test_editing_the_script_would_change_the_key(self, tmp_path: Path) -> None:
        """This is the point of content addressing: a stable key can serve stale bytes."""
        import hashlib

        original = bootstrap_text()
        mutated = original + "\n# a change\n"
        assert (
            hashlib.sha256(original.encode()).hexdigest()[:8]
            != (hashlib.sha256(mutated.encode()).hexdigest()[:8])
        )


class TestUploadCommand:
    def test_is_an_aws_s3_cp_invocation(self, settings: Settings) -> None:
        argv = upload_argv(settings)
        assert argv[:3] == ["aws", "s3", "cp"]

    def test_source_is_the_packaged_script(self, settings: Settings) -> None:
        assert upload_argv(settings)[3] == str(bootstrap_path())

    def test_destination_is_the_content_addressed_url(self, settings: Settings) -> None:
        assert upload_argv(settings)[4] == bootstrap_s3_url(settings)

    def test_includes_the_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        argv = upload_argv(load_settings(live=False))
        assert argv[argv.index("--region") + 1] == "eu-west-1"
