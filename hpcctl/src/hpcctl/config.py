"""Environment-variable contract and resolved settings.

This module owns *all* environment access. No other module reads ``os.environ`` -- that is what
makes the resolved configuration of an invocation a single testable value rather than something
scattered across five commands.

Missing required variables behave differently by mode, and the asymmetry is deliberate:

* **Dry-run** substitutes ``<<<UNSET:HPCCTL_KEY_NAME>>>``, records it, warns, and continues.
  Requiring a real subnet ID merely to *print* a config would make the tool unusable until the
  AWS account exists, and would make CI impossible.
* **Live** fails before the first API call, listing every missing variable at once. A live run
  that discovers a missing variable halfway through has already created billable resources.

The placeholder format earns its ugliness: ``<<<UNSET:NAME>>>`` is a legal YAML scalar, so
generated config still parses and the "YAML parses" test stays meaningful, but no AWS API would
ever accept it, so a placeholder can never be mistaken for a real value or silently work.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from hpcctl.errors import InvalidConfigError, MissingConfigError

PLACEHOLDER_FORMAT: Final[str] = "<<<UNSET:{name}>>>"
"""Substituted for a required variable that was unset during a dry-run."""

REQUIRED_ALWAYS: Final[frozenset[str]] = frozenset(
    {
        "AWS_REGION",
        "HPCCTL_KEY_NAME",
        "HPCCTL_HEAD_SUBNET_ID",
        "HPCCTL_BOOTSTRAP_BUCKET",
    }
)
"""Variables a live cluster-creating command cannot proceed without."""

REQUIRED_FOR_REMOTE: Final[frozenset[str]] = frozenset({"HPCCTL_HEAD_NODE_HOST"})
"""Additionally required by commands that reach the head node over SSH."""

REQUIRED_FOR_CLUSTER: Final[frozenset[str]] = frozenset({"AWS_REGION"})
"""Required by commands that only query or delete an existing cluster."""


def placeholder(name: str) -> str:
    """Render the placeholder standing in for an unset variable.

    Args:
        name: Environment variable name.

    Returns:
        For example ``"<<<UNSET:AWS_REGION>>>"``.
    """
    return PLACEHOLDER_FORMAT.format(name=name)


def is_placeholder(value: str) -> bool:
    """Report whether a resolved value is a substituted placeholder.

    Args:
        value: A resolved settings value.

    Returns:
        ``True`` if the value was substituted for an unset variable.
    """
    return value.startswith("<<<UNSET:") and value.endswith(">>>")


def dry_run_forced() -> bool:
    """Report whether ``HPCCTL_DRY_RUN`` is set to a non-empty value.

    A global kill switch that a flag cannot override is worth more than flag-precedence purity
    while the AWS account does not exist.

    Returns:
        ``True`` when live execution must be refused regardless of ``--execute``.
    """
    return bool(os.environ.get("HPCCTL_DRY_RUN", ""))


def color_disabled() -> bool:
    """Report whether the conventional ``NO_COLOR`` variable is set.

    Returns:
        ``True`` when colour output should be suppressed.
    """
    return bool(os.environ.get("NO_COLOR", ""))


def default_schema_path() -> Path:
    """Locate the repository's DAG schema by walking up from this file.

    Returns:
        The path to ``shared/dag_schema.json`` if found, otherwise the relative fallback
        ``shared/dag_schema.json`` so the error names something recognisable.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "shared" / "dag_schema.json"
        if candidate.is_file():
            return candidate
    return Path("shared") / "dag_schema.json"


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one hpcctl invocation.

    Attributes:
        cluster_name: ParallelCluster cluster name.
        region: AWS region.
        os_image: ParallelCluster ``Image.Os`` value.
        key_name: EC2 key pair *name*, not a path.
        head_subnet_id: Subnet for the head node.
        compute_subnet_id: Subnet for compute nodes; defaults to the head subnet.
        head_instance_type: EC2 instance type for the head node.
        compute_instance_type: EC2 instance type for compute nodes.
        queue_name: Slurm queue (partition) name.
        min_nodes: Minimum compute nodes in the queue.
        max_nodes: Maximum compute nodes in the queue.
        shared_dir: Mount point of the shared filesystem.
        shared_volume_gb: Size of the shared EBS volume in GiB.
        bootstrap_bucket: S3 bucket the bootstrap script is published to.
        bootstrap_prefix: S3 key prefix for the bootstrap script.
        head_node_host: Hostname or IP of the head node.
        ssh_user: SSH login user on cluster nodes.
        ssh_key_path: Path to the private key. Its *contents* are never printed.
        engine_build_dir: Local directory holding compiled engine binaries.
        remote_engine_dir: Remote directory binaries are synced to.
        remote_dag_dir: Remote directory serialized DAGs are staged in.
        engine_binary: Remote path of the engine executable.
        ntasks: Default Slurm task count.
        nodes: Default Slurm node count.
        time_limit: Default Slurm wall-clock limit.
        schema_path: Path to the DAG serialization contract.
        run_dir: Directory generated artifacts are written to.
        missing: Names of required variables that were unset and replaced with placeholders.
            Non-empty only in dry-run; live resolution raises instead.
    """

    cluster_name: str
    region: str
    os_image: str
    key_name: str
    head_subnet_id: str
    compute_subnet_id: str
    head_instance_type: str
    compute_instance_type: str
    queue_name: str
    min_nodes: int
    max_nodes: int
    shared_dir: str
    shared_volume_gb: int
    bootstrap_bucket: str
    bootstrap_prefix: str
    head_node_host: str
    ssh_user: str
    ssh_key_path: str
    engine_build_dir: Path
    remote_engine_dir: str
    remote_dag_dir: str
    engine_binary: str
    ntasks: int
    nodes: int
    time_limit: str
    schema_path: Path
    run_dir: Path
    missing: tuple[str, ...]

    @property
    def has_placeholders(self) -> bool:
        """Return whether any required value was substituted with a placeholder."""
        return bool(self.missing)


def _env(name: str, default: str) -> str:
    """Read a variable that has a default, treating empty as unset.

    Args:
        name: Environment variable name.
        default: Value to use when unset or empty.

    Returns:
        The resolved string value.
    """
    return os.environ.get(name, "").strip() or default


def _env_int(name: str, default: int) -> int:
    """Read an integer-valued variable.

    Args:
        name: Environment variable name.
        default: Value to use when unset or empty.

    Returns:
        The parsed integer.

    Raises:
        InvalidConfigError: If the value is set but not a base-10 integer.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise InvalidConfigError(
            f"{name} must be an integer, got {raw!r}",
            hint=f"Unset {name} to use the default of {default}.",
        ) from exc


def load_settings(
    *,
    live: bool,
    strict: bool = False,
    required: frozenset[str] = REQUIRED_ALWAYS,
) -> Settings:
    """Resolve configuration from the environment.

    Args:
        live: Whether the caller intends to execute against AWS. When true, any missing
            required variable raises immediately, before a single API call is made.
        strict: Treat missing required variables as fatal even when ``live`` is false.
            Intended for CI that wants to verify a fully-specified config.
        required: Variable names this command cannot run live without. Defaults to the
            cluster-creation set; commands needing SSH add :data:`REQUIRED_FOR_REMOTE`.

    Returns:
        Fully resolved settings. In dry-run, missing required values are the string
        ``"<<<UNSET:VAR_NAME>>>"`` and are listed in ``Settings.missing``.

    Raises:
        MissingConfigError: If a required variable is unset and ``live`` or ``strict`` is set.
        InvalidConfigError: If a numeric variable is set to a non-integer.
    """
    missing: list[str] = []

    def optional(name: str, fallbacks: tuple[str, ...] = ()) -> str:
        """Read a variable that has no default, recording it when unset.

        Args:
            name: Primary environment variable name.
            fallbacks: Alternative names consulted in order before giving up.

        Returns:
            The resolved value, or a placeholder.
        """
        for candidate in (name, *fallbacks):
            value = os.environ.get(candidate, "").strip()
            if value:
                return value
        missing.append(name)
        return placeholder(name)

    region = optional("AWS_REGION", ("AWS_DEFAULT_REGION",))
    key_name = optional("HPCCTL_KEY_NAME")
    head_subnet_id = optional("HPCCTL_HEAD_SUBNET_ID")
    bootstrap_bucket = optional("HPCCTL_BOOTSTRAP_BUCKET")
    head_node_host = optional("HPCCTL_HEAD_NODE_HOST")

    # Inherits the head subnet, including its placeholder, so a single-subnet VPC needs one
    # variable rather than two.
    compute_subnet_id = _env("HPCCTL_COMPUTE_SUBNET_ID", head_subnet_id)

    shared_dir = _env("HPCCTL_SHARED_DIR", "/shared")
    remote_engine_dir = _env("HPCCTL_REMOTE_ENGINE_DIR", f"{shared_dir}/engine")
    remote_dag_dir = _env("HPCCTL_REMOTE_DAG_DIR", f"{shared_dir}/dags")
    engine_binary = _env("HPCCTL_ENGINE_BINARY", f"{remote_engine_dir}/bin/engine")

    schema_override = os.environ.get("HPCCTL_SCHEMA_PATH", "").strip()
    schema_path = Path(schema_override) if schema_override else default_schema_path()

    blocking = sorted(name for name in missing if name in required)
    if blocking and (live or strict):
        raise MissingConfigError(
            blocking,
            hint=(
                "Set them in the environment (see hpcctl/.env.example), or drop --execute "
                "to preview with placeholders."
            )
            if live
            else "Set them in the environment, or drop --strict.",
        )

    return Settings(
        cluster_name=_env("HPCCTL_CLUSTER_NAME", "hpc-dag-baseline"),
        region=region,
        os_image=_env("HPCCTL_OS", "ubuntu2204"),
        key_name=key_name,
        head_subnet_id=head_subnet_id,
        compute_subnet_id=compute_subnet_id,
        head_instance_type=_env("HPCCTL_HEAD_INSTANCE_TYPE", "t3.medium"),
        compute_instance_type=_env("HPCCTL_COMPUTE_INSTANCE_TYPE", "c5.large"),
        queue_name=_env("HPCCTL_QUEUE_NAME", "compute"),
        min_nodes=_env_int("HPCCTL_MIN_NODES", 0),
        max_nodes=_env_int("HPCCTL_MAX_NODES", 4),
        shared_dir=shared_dir,
        shared_volume_gb=_env_int("HPCCTL_SHARED_VOLUME_GB", 50),
        bootstrap_bucket=bootstrap_bucket,
        bootstrap_prefix=_env("HPCCTL_BOOTSTRAP_PREFIX", "hpcctl/bootstrap"),
        head_node_host=head_node_host,
        ssh_user=_env("HPCCTL_SSH_USER", "ubuntu"),
        ssh_key_path=str(Path(_env("HPCCTL_SSH_KEY_PATH", "~/.ssh/id_rsa")).expanduser()),
        engine_build_dir=Path(_env("HPCCTL_ENGINE_BUILD_DIR", "./engine/build")),
        remote_engine_dir=remote_engine_dir,
        remote_dag_dir=remote_dag_dir,
        engine_binary=engine_binary,
        ntasks=_env_int("HPCCTL_NTASKS", 4),
        nodes=_env_int("HPCCTL_NODES", 2),
        time_limit=_env("HPCCTL_TIME_LIMIT", "00:30:00"),
        schema_path=schema_path,
        run_dir=Path(_env("HPCCTL_RUN_DIR", "./.hpcctl-run")),
        missing=tuple(missing),
    )
