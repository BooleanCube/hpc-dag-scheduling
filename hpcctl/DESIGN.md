# `/hpcctl` Cluster CLI — Interface Design

Implementation-ready design for the AWS ParallelCluster management CLI. Target: Python 3.11,
`mypy --strict`, `ruff` (rules `E,F,I,N,UP,B,D`, line length 100, Google docstring convention),
Typer 0.27.1.

Two documents constrain this one: [`/shared/dag_schema.json`](../shared/dag_schema.json) is the DAG
contract `submit` validates against, and the repo-root `.gitignore` dictates where generated
artifacts may land (§10). Where this document disagrees with either, they win.

---

## 1. The governing constraint: no AWS credits

There is no AWS account behind this yet, and the dev box has neither tool installed:

```
pcluster: NOT installed        aws: NOT installed
rsync: /usr/bin/rsync         sbatch: NOT installed (expected on dev VM)
```

That is not a temporary inconvenience to work around — it is the design's central input. It produces
four principles that the rest of the document follows mechanically.

**P1 — Dry-run must run fully offline.** Every dry-run path must complete successfully with no AWS
credentials, no `pcluster`, no `aws` CLI, no network, and no environment variables set. If a dry-run
needs any of those, it cannot be exercised today and cannot be tested in CI, which defeats the
purpose. This is why missing env vars become placeholders in dry-run rather than errors (§4).

**P2 — Dry-run is the DEFAULT; live execution is opt-in.** Every AWS-touching command defaults to
`--dry-run` and requires an explicit `--execute` to do anything real.

> **Decision for team-lead.** CLAUDE.md §2 requires that AWS commands *support* `--dry-run`; it does
> not say which way the default points. I am recommending default-on because the failure modes are
> asymmetric: a forgotten flag on a default-live CLI spends money the project does not have or leaves
> orphaned EC2 instances billing, while a forgotten flag on a default-dry CLI prints a plan and wastes
> ten seconds. Both flags exist, so the mandate is satisfied either way. If you prefer default-live,
> the only change is the default value of one Typer option per command.

**P3 — Generators return strings; the console is a separate layer.** Every artifact (YAML, bash,
sbatch) is produced by a pure function returning `str`. Rendering is a distinct, lossy presentation
step (§9). Tests call the generators directly and never scrape CLI stdout. This is not stylistic —
§9 documents a measured case where the display layer silently drops characters.

**P4 — The CLI validates the contract, not the builder.** `hpcctl` does **not** depend on the
`tasks` package. `submit` validates a serialized DAG file against `/shared/dag_schema.json` with
`jsonschema`, exactly as the C++ engine will. Importing `tasks` would drag NumPy and the whole
builder into a tool whose job is to manage EC2 instances, and would couple two projects that the
`/shared` contract exists specifically to decouple. If `tasks` happens to be importable, do not use
it; one validation path means one set of error messages.

---

## 2. Dependency decisions

Current declared dependency is `typer>=0.27.1` only. Add three, remove none:

| Package | Group | Why |
| --- | --- | --- |
| `rich` | runtime | Panels, syntax highlighting, status tables. |
| `pyyaml` | runtime | `yaml.safe_dump` for the cluster config; `safe_load` in tests. |
| `jsonschema` | runtime | Validates DAG files against the draft 2020-12 contract. |
| `types-PyYAML` | dev | `mypy --strict` has no bundled stubs for `yaml`. |

```bash
uv add rich pyyaml jsonschema
uv add --dev types-PyYAML
```

**Declare `rich` explicitly even though it is already installed.** Typer 0.27.1 lists
`rich>=13.8.0` among its own requirements, so `import rich` works today by accident. Depending on
another package's transitive dependency is exactly the kind of thing that breaks on an unrelated
upgrade. We import it directly, so we declare it directly.

**Do NOT add `aws-parallelcluster` or `awscli` as dependencies.** We shell out to `pcluster` and
`aws` rather than importing them. Adding them would pull a large transitive tree into a CLI whose
dry-run path must work with neither installed (P1), and it would make `uv sync` slow for no benefit.
`external.py` (§7) discovers them at call time and exits 5 with an actionable message when a live
command needs a missing tool. Note that `click` is **not** a Typer 0.27 dependency any more —
verified absent from the venv — so never import `click` directly.

---

## 3. Module layout

```
hpcctl/src/hpcctl/
├── __init__.py              # exports main() (exists)
├── cli.py                   # root Typer app; registers command modules (exists)
├── exit_codes.py            # ExitCode IntEnum
├── errors.py                # HpcctlError hierarchy, each carrying an ExitCode
├── config.py                # Settings, env loading, placeholder machinery
├── console.py               # rich Console + lossless artifact rendering
├── external.py              # tool discovery + subprocess wrapper (dry-run aware)
├── validation.py            # DAG file -> schema validation
├── generators/
│   ├── __init__.py
│   ├── cluster_config.py    # Settings -> dict -> yaml.safe_dump
│   ├── sbatch.py            # Settings + DAG path -> sbatch text
│   └── bootstrap.py         # locate + hash the packaged bootstrap script
├── bootstrap/
│   └── install_engine_deps.sh   # committed, dual-use (§6)
└── commands/
    ├── __init__.py
    ├── boot.py
    ├── deploy.py
    ├── submit.py
    ├── status.py
    └── destroy.py
```

**The bootstrap script lives inside the package**, at `src/hpcctl/bootstrap/install_engine_deps.sh`,
not at the project root. I verified that `uv_build` ships non-Python files under the module directory
without any extra configuration — a test wheel built from an equivalent layout contained
`demo/bootstrap/install.sh` — so the script travels with an installed `hpcctl` and is locatable via
`importlib.resources.files("hpcctl") / "bootstrap" / "install_engine_deps.sh"`. Resolve it that way
rather than by walking up from `__file__`, so the CLI works when installed as a wheel and not only
from a source checkout. A `.sh` file is not touched by any `.gitignore` rule, so it commits cleanly.

---

## 4. Configuration: the environment variable contract

`config.py` owns all environment access. No other module reads `os.environ`.

```python
PLACEHOLDER_FORMAT = "<<<UNSET:{name}>>>"


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one hpcctl invocation.

    Attributes:
        missing: Names of required variables that were unset and replaced with placeholders.
            Non-empty only in dry-run; live resolution raises instead.
    """

    cluster_name: str
    region: str
    # ... one field per row of the table below
    missing: tuple[str, ...]

    @property
    def has_placeholders(self) -> bool:
        """Return whether any required value was substituted with a placeholder."""


def load_settings(*, live: bool, strict: bool = False) -> Settings:
    """Resolve configuration from the environment.

    Args:
        live: Whether the caller intends to execute against AWS. When true, any missing
            required variable raises immediately, before a single API call is made.
        strict: Treat missing required variables as fatal even when ``live`` is false.
            Intended for CI that wants to verify a fully-specified config.

    Returns:
        Fully resolved settings. In dry-run, missing required values are the string
        ``"<<<UNSET:VAR_NAME>>>"`` and are listed in ``Settings.missing``.

    Raises:
        MissingConfigError: If a required variable is unset and ``live`` or ``strict`` is set.
    """
```

### Variable table

R = required for live execution. All are read only from the environment; none may be hardcoded or
committed, per CLAUDE.md §4.

| Variable | R | Default | Used by |
| --- | --- | --- | --- |
| `HPCCTL_CLUSTER_NAME` | | `hpc-dag-baseline` | boot, destroy, status, deploy, submit |
| `AWS_REGION` | R | — (falls back to `AWS_DEFAULT_REGION`) | boot, destroy, status |
| `HPCCTL_OS` | | `ubuntu2204` | boot |
| `HPCCTL_KEY_NAME` | R | — | boot (EC2 key pair *name*, not a path) |
| `HPCCTL_HEAD_SUBNET_ID` | R | — | boot |
| `HPCCTL_COMPUTE_SUBNET_ID` | | falls back to `HPCCTL_HEAD_SUBNET_ID` | boot |
| `HPCCTL_HEAD_INSTANCE_TYPE` | | `t3.medium` | boot |
| `HPCCTL_COMPUTE_INSTANCE_TYPE` | | `c5.large` | boot |
| `HPCCTL_QUEUE_NAME` | | `compute` | boot, submit |
| `HPCCTL_MIN_NODES` | | `0` | boot |
| `HPCCTL_MAX_NODES` | | `4` | boot |
| `HPCCTL_SHARED_DIR` | | `/shared` | boot, deploy, submit |
| `HPCCTL_SHARED_VOLUME_GB` | | `50` | boot |
| `HPCCTL_BOOTSTRAP_BUCKET` | R | — | boot (S3 upload target) |
| `HPCCTL_BOOTSTRAP_PREFIX` | | `hpcctl/bootstrap` | boot |
| `HPCCTL_HEAD_NODE_HOST` | R for deploy/submit/status | — (else discovered, §5) | deploy, submit, status |
| `HPCCTL_SSH_USER` | | `ubuntu` | deploy, submit, status |
| `HPCCTL_SSH_KEY_PATH` | R for deploy/submit/status | `~/.ssh/id_rsa` | deploy, submit, status |
| `HPCCTL_ENGINE_BUILD_DIR` | | `./engine/build` | deploy |
| `HPCCTL_REMOTE_ENGINE_DIR` | | `${HPCCTL_SHARED_DIR}/engine` | deploy, submit |
| `HPCCTL_REMOTE_DAG_DIR` | | `${HPCCTL_SHARED_DIR}/dags` | submit |
| `HPCCTL_ENGINE_BINARY` | | `${HPCCTL_REMOTE_ENGINE_DIR}/bin/engine` | submit |
| `HPCCTL_NTASKS` | | `4` | submit |
| `HPCCTL_NODES` | | `2` | submit |
| `HPCCTL_TIME_LIMIT` | | `00:30:00` | submit |
| `HPCCTL_SCHEMA_PATH` | | repo `shared/dag_schema.json` | submit |
| `HPCCTL_RUN_DIR` | | `./.hpcctl-run` | all (generated artifacts, §10) |
| `HPCCTL_DRY_RUN` | | unset | global override, forces dry-run everywhere |

Ship a committed `hpcctl/.env.example` enumerating every row with placeholder values. The root
`.gitignore` ignores `.env` and `.env.*` but has an explicit `!.env.example` negation, so this is the
one config artifact that can be tracked — and it replaces the YAML template we are forbidden from
committing (§10). `hpcctl` does **not** auto-load `.env`; the user sources it. Auto-loading hidden
files makes the resolved config non-obvious, which is a bad property for a tool that spends money.

### Missing-variable behaviour, and why it differs by mode

**Dry-run: substitute `<<<UNSET:HPCCTL_KEY_NAME>>>`, warn, continue, exit 0.**
**Live: fail before the first API call, exit 3, listing every missing variable at once.**

The asymmetry is the point of P1. Requiring a real subnet ID to *print* a config would make the tool
unusable until the account exists, and would make CI impossible. Conversely, a live run that
discovers a missing variable halfway through has already created billable resources, so it must
validate everything up front and report all failures in one pass rather than one per run.

The placeholder format earns its ugliness. `<<<UNSET:NAME>>>` is a legal YAML scalar, so generated
config still parses and the "YAML parses" test stays meaningful; but no AWS API would ever accept it,
so a placeholder can never be mistaken for a real value or silently work. Dry-run output ends with a
summary panel listing the substituted variables, and `--strict` promotes them to a hard failure for
CI that wants to verify a complete config.

---

## 5. Commands

Registered on the existing `app` in `cli.py` via `app.command()(...)` from each module in
`commands/`. The root callback gains two global options: `--no-color` and `-v/--verbose`.

Every AWS-touching command carries this identical pair, which belongs in a shared Typer annotation
so the wording cannot drift:

```python
dry_run: Annotated[
    bool,
    typer.Option(
        "--dry-run/--execute",
        help="Print intended actions instead of performing them. Default: dry-run.",
    ),
] = True
```

`HPCCTL_DRY_RUN` set to anything non-empty forces dry-run even when `--execute` is passed, and says
so on stderr. A global kill switch that cannot be overridden by a flag is worth more than
flag-precedence purity while the account does not exist.

| Command | AWS-touching | Needs `--dry-run` | Local-only fallback |
| --- | --- | --- | --- |
| `boot` | yes (S3 upload + `pcluster create-cluster`) | yes | full artifact generation |
| `deploy` | yes (SSH/rsync to head node) | yes | — |
| `submit` | partly | yes | `--validate-only` is fully local |
| `status` | yes (`pcluster describe-cluster` + SSH `squeue`) | yes | — |
| `destroy` | yes (`pcluster delete-cluster`) | yes | — |
| `version` | no | no | already implemented |

### `hpcctl boot`

```python
def boot(
    dry_run: bool = True,
    strict: bool = False,
    emit_dir: Annotated[
        Path | None, typer.Option(help="Write artifacts here instead of RUN_DIR.")
    ] = None,
    raw: Annotated[bool, typer.Option(help="Print artifacts unformatted, for piping.")] = False,
) -> None:
    """Create the ParallelCluster cluster.

    In dry-run this prints the three artifacts that would be used -- the generated
    ParallelCluster YAML, the Ubuntu bootstrap script that nodes will execute, and the exact
    ``pcluster create-cluster`` invocation -- and writes them to the run directory.
    """
```

Dry-run prints exactly the three artifacts requested, in dependency order so the output reads as a
narrative rather than a dump:

1. **The bootstrap script** (§6) with its SHA-256, plus the `aws s3 cp` command that would upload it.
2. **The cluster YAML** (§8), whose `OnNodeConfigured.Script` points at the content-addressed S3 URL
   from step 1.
3. **The `pcluster create-cluster` command**, referencing the config path from step 2.

Live sequence, in this order, because each step is a precondition for the next:

1. `load_settings(live=True)` — fail fast, exit 3.
2. Discover `aws` and `pcluster` — exit 5 if absent.
3. `aws s3 cp` the bootstrap script to its content-addressed key.
4. Write the config to the run directory.
5. `pcluster create-cluster --cluster-name … --cluster-configuration …`.
6. Print the returned cluster status; note that creation is asynchronous and point at
   `hpcctl status`.

### `hpcctl deploy`

```python
def deploy(dry_run: bool = True, build_dir: Path | None = None, strict: bool = False) -> None:
    """Sync compiled engine binaries to the cluster's shared filesystem."""
```

Target is `HPCCTL_REMOTE_ENGINE_DIR`, which defaults under `HPCCTL_SHARED_DIR` (`/shared`) rather
than the head node's home directory. This matters for correctness, not tidiness: compute nodes must
see the same binary the head node has, so `deploy` and the cluster's `SharedStorage` mount point
(§8) must agree. Deploying to `~ubuntu` would produce jobs that run on the head node and fail on
every compute node with "No such file or directory".

Dry-run prints the resolved `rsync` command and a manifest table of files that would transfer (name,
size, mtime) — obtained locally with `rsync --dry-run --itemize-changes` against the *local* build
dir, so no SSH is needed. Verify the local build dir exists and is non-empty even in dry-run; that is
a local check and catching "you have not built the engine yet" without an AWS account is free.

```
rsync -avz --delete -e "ssh -i <key> -o StrictHostKeyChecking=accept-new" \
  ./engine/build/ ubuntu@<host>:/shared/engine/
```

`StrictHostKeyChecking=accept-new` rather than `no`: it still pins the key after first contact, so it
protects against later MITM while not prompting on first connect. Never `no`. Never
`UserKnownHostsFile=/dev/null`. `/engine` is read-only to us — `deploy` only ever reads from
`HPCCTL_ENGINE_BUILD_DIR` and must never write into the engine tree.

### `hpcctl submit`

```python
def submit(
    dag: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    dry_run: bool = True,
    validate_only: bool = False,
    job_name: str | None = None,
    nodes: int | None = None,
    ntasks: int | None = None,
    time_limit: str | None = None,
) -> None:
    """Validate a serialized DAG and submit it to Slurm as a batch job."""
```

Order is deliberate — validate first, always, in every mode:

1. Parse the DAG file; malformed JSON exits 4 with the line and column from
   `json.JSONDecodeError`.
2. Validate against the schema (§11). On failure, print every error as a table (JSON pointer path,
   message) and exit 4. Use `sorted(validator.iter_errors(doc), key=str)` so multiple problems are
   reported at once and in a stable order.
3. If `--validate-only`, exit 0 here. This path touches nothing but the filesystem and is the
   fastest useful thing the CLI can do today.
4. Generate the sbatch script (§8) and write it to the run directory.
5. Dry-run: print the sbatch script, plus the `scp` and `ssh … sbatch` commands. Live: `scp` the DAG
   and the script to the head node, run `ssh … sbatch`, parse the job ID out of
   `Submitted batch job (\d+)`, and print it.

CLI options override their env-var defaults, since job geometry is the thing a user tunes per
experiment.

### `hpcctl status`

```python
def status(dry_run: bool = True, queue: bool = True, watch: bool = False) -> None:
    """Report cluster and Slurm queue status."""
```

Two independent rich tables: cluster (name, status, region, head node IP, compute fleet) from
`pcluster describe-cluster`, and the Slurm queue (job ID, name, state, nodes, elapsed) from
`ssh … squeue`. Degrade rather than fail: if the cluster query succeeds but SSH does not, print the
cluster table plus a warning and exit 0 — a cluster that is still creating has no reachable head node
yet, and that is normal, not an error. Exit 8 only when the cluster itself is absent or in a failed
state. Dry-run prints both commands and a table of placeholder rows so the layout is reviewable.

`--watch` re-renders on an interval with `rich.live.Live`; forbid it in dry-run, where it would loop
over static text forever.

### `hpcctl destroy`

```python
def destroy(dry_run: bool = True, yes: bool = False) -> None:
    """Delete the cluster and all associated compute resources."""
```

Confirmation UX: require the user to **type the cluster name**, the way GitHub gates repository
deletion. A `y/N` prompt is one keystroke from destroying a running experiment, and the cluster name
is the one thing a user who means it can always produce. Mismatch exits 7.

```
About to DELETE cluster 'hpc-dag-baseline' in us-east-1.
This terminates the head node and all compute nodes. Running jobs will be lost.
Type the cluster name to confirm:
```

Three rules on top: `--yes` skips the prompt for automation; if stdin is not a TTY and `--yes` was
not passed, exit 7 without prompting rather than blocking forever on a pipe; and **dry-run never
prompts at all** — it prints the `pcluster delete-cluster` command and exits, since there is nothing
to confirm. Prompting in dry-run would train users to type the confirmation reflexively, which is
precisely the habit this UX exists to prevent.

### Exit codes

`exit_codes.py`:

```python
class ExitCode(IntEnum):
    """Process exit statuses. Stable contract for CI and shell scripts."""

    OK = 0
    INTERNAL = 1  # unexpected exception; always a bug
    USAGE = 2  # reserved: Typer/argument parsing
    CONFIG = 3  # required env var missing or malformed
    DAG_INVALID = 4  # DAG file failed schema validation
    TOOL_MISSING = 5  # pcluster/aws/ssh/rsync not on PATH
    COMMAND_FAILED = 6  # external command returned non-zero
    ABORTED = 7  # user declined confirmation, or no TTY for one
    CLUSTER_STATE = 8  # cluster absent or in an unexpected state
```

2 is reserved rather than assigned because Typer already exits 2 on argument errors; claiming it
would create two meanings for one code. `errors.py` defines `HpcctlError(Exception)` with a
`code: ExitCode` attribute and one subclass per non-trivial code (`MissingConfigError`,
`DagValidationError`, `ToolMissingError`, `ExternalCommandError`, `AbortedError`,
`ClusterStateError`). `cli.py` wraps invocation in a single handler that catches `HpcctlError`,
prints `err.message` to stderr, and raises `typer.Exit(err.code)`. Individual commands never call
`sys.exit`; they raise. One exit path is testable, five are not.

---

## 6. The bootstrap script

Path: `hpcctl/src/hpcctl/bootstrap/install_engine_deps.sh`. Committed, executable, dual-use.

Two callers, one file:

```bash
# On the dev VM, by hand:
bash hpcctl/src/hpcctl/bootstrap/install_engine_deps.sh

# On every cluster node, automatically, as root, via ParallelCluster:
CustomActions:
  OnNodeConfigured:
    Script: s3://<bucket>/hpcctl/bootstrap/install_engine_deps-<sha8>.sh
```

The non-negotiables from the brief plus the ones that specifically prevent a hung node boot:

- `#!/usr/bin/env bash` and `set -euo pipefail`.
- `export DEBIAN_FRONTEND=noninteractive` before any apt invocation.
- `apt-get install -y`, never bare `apt`/`apt install`. `apt` is a human-facing wrapper that emits
  warnings and can change behaviour between releases; `apt-get` is the stable scripting interface.
- `-o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold`. **This is the one most
  likely to be omitted and the most damaging.** `DEBIAN_FRONTEND=noninteractive` suppresses debconf
  prompts but *not* dpkg's "a config file has been modified, keep or replace?" prompt, which will
  hang node configuration until the ParallelCluster timeout fires and the node is marked failed.
- `export NEEDRESTART_MODE=a` and `NEEDRESTART_SUSPEND=1`. On Ubuntu 22.04 `needrestart` can
  interject during apt transactions; both are cheap insurance.
- Idempotent, via a version-stamped marker at `/var/lib/hpcctl/bootstrap.v1.done`. Bumping the
  embedded `BOOTSTRAP_VERSION` invalidates the marker and forces a genuine re-run, so the marker
  never masks an updated package list.
- `apt-get update` in a retry loop. Transient mirror failures during simultaneous boot of a whole
  compute fleet are common, and one failed `update` under `set -e` kills the node.
- Never `apt-get upgrade` or `dist-upgrade`. Unbounded runtime and unrelated changes during node
  boot is exactly what you do not want between you and a running job.
- Works as root (cluster) and as a normal user (dev VM) via a `SUDO` shim.
- Ends with a verification block that fails loudly if a required tool is missing, so a node that
  cannot build the engine fails at configuration time rather than at first job.

### Package list

Required: `build-essential`, `cmake`, `ninja-build`, `git`, `pkg-config`, `ca-certificates`, `curl`,
`unzip`, `openmpi-bin`, `libopenmpi-dev`, `nlohmann-json3-dev`, `libprotobuf-dev`,
`protobuf-compiler`.

Optional, behind `--with-dev-tools`, off by default for node boot: `gdb`, `valgrind`, `clang-format`,
`clang-tidy`.

> **Risk to verify when credits exist.** ParallelCluster's Ubuntu AMIs ship their own MPI stack
> (commonly Open MPI plus Intel MPI, exposed through environment modules). Installing
> `libopenmpi-dev` from the Ubuntu archive may shadow it and produce a binary that links against a
> different MPI than `srun --mpi=pmix` expects — which surfaces as a rank-0-only run or an MPI init
> failure, not as a build error. The first live `boot` should run `mpicc --show` and `module avail`
> on a compute node and compare. If they conflict, prefer the AMI's MPI and drop the two `openmpi`
> packages. I cannot settle this without an account, so it is flagged rather than guessed.

### Script

Builder should copy this verbatim; it is written to pass `bash -n` and `shellcheck` as-is.

```bash
#!/usr/bin/env bash
#
# Install the C++ engine's Ubuntu build dependencies.
#
# Dual-use:
#   * directly on a dev VM:  bash install_engine_deps.sh
#   * on cluster nodes:      ParallelCluster CustomActions/OnNodeConfigured (runs as root)
#
# Must never prompt: an interactive prompt here hangs node configuration until the
# ParallelCluster timeout fires and the node is marked failed.

set -euo pipefail

BOOTSTRAP_VERSION="1"
MARKER="/var/lib/hpcctl/bootstrap.v${BOOTSTRAP_VERSION}.done"
WITH_DEV_TOOLS="no"

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

APT_OPTS=(
  -y
  -o Dpkg::Options::=--force-confdef
  -o Dpkg::Options::=--force-confold
)

PACKAGES=(
  build-essential
  cmake
  ninja-build
  git
  pkg-config
  ca-certificates
  curl
  unzip
  openmpi-bin
  libopenmpi-dev
  nlohmann-json3-dev
  libprotobuf-dev
  protobuf-compiler
)

DEV_PACKAGES=(
  gdb
  valgrind
  clang-format
  clang-tidy
)

log() {
  printf '[hpcctl-bootstrap] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Usage: install_engine_deps.sh [--with-dev-tools] [--force] [--help]

  --with-dev-tools  Also install gdb, valgrind, clang-format, clang-tidy.
  --force           Reinstall even if the completion marker is present.
  --help            Show this message.
USAGE
}

FORCE="no"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-dev-tools) WITH_DEV_TOOLS="yes" ;;
    --force) FORCE="yes" ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
  shift
done

# Root on cluster nodes, unprivileged on a dev VM.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  command -v sudo >/dev/null 2>&1 || die "not root and sudo is unavailable"
  SUDO="sudo"
fi

if [ "$FORCE" = "no" ] && [ -f "$MARKER" ]; then
  log "marker $MARKER present; already provisioned (use --force to override)"
  exit 0
fi

# Mirror flakiness during a simultaneous fleet boot is routine; one failure under
# set -e would otherwise kill the node.
apt_update_with_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if $SUDO apt-get update; then
      return 0
    fi
    log "apt-get update failed (attempt ${attempt}/5); retrying in $((attempt * 5))s"
    sleep "$((attempt * 5))"
  done
  die "apt-get update failed after 5 attempts"
}

log "starting bootstrap version ${BOOTSTRAP_VERSION} on $(. /etc/os-release && echo "${PRETTY_NAME}")"
apt_update_with_retry

log "installing ${#PACKAGES[@]} required packages"
$SUDO apt-get install "${APT_OPTS[@]}" "${PACKAGES[@]}"

if [ "$WITH_DEV_TOOLS" = "yes" ]; then
  log "installing ${#DEV_PACKAGES[@]} optional dev packages"
  $SUDO apt-get install "${APT_OPTS[@]}" "${DEV_PACKAGES[@]}"
fi

# Fail at configuration time rather than at first job.
log "verifying toolchain"
MISSING=""
for tool in gcc g++ cmake ninja git mpicc mpirun protoc; do
  command -v "$tool" >/dev/null 2>&1 || MISSING="${MISSING} ${tool}"
done
[ -z "$MISSING" ] || die "missing after install:${MISSING}"

# Header-only, so presence is a file check rather than a command check.
NLOHMANN_HEADER="/usr/include/nlohmann/json.hpp"
[ -f "$NLOHMANN_HEADER" ] || die "missing header: ${NLOHMANN_HEADER}"

log "cmake:  $(cmake --version | head -n1)"
log "gcc:    $(gcc --version | head -n1)"
log "mpicc:  $(mpicc --version 2>/dev/null | head -n1 || echo 'version unavailable')"
log "protoc: $(protoc --version)"

$SUDO mkdir -p "$(dirname "$MARKER")"
printf 'version=%s\ncompleted=%s\n' \
  "$BOOTSTRAP_VERSION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  | $SUDO tee "$MARKER" >/dev/null

log "bootstrap complete"
```

### Getting the script to the nodes

`OnNodeConfigured` takes an `s3://` or `https://` URL, not a local path, so `boot` must publish it.
The mechanism is **content addressing**:

1. Read the packaged script via `importlib.resources`.
2. `sha256` it; take the first 8 hex characters.
3. Key: `s3://${HPCCTL_BOOTSTRAP_BUCKET}/${HPCCTL_BOOTSTRAP_PREFIX}/install_engine_deps-<sha8>.sh`.
4. Upload with `aws s3 cp`.
5. Reference that exact URL in the generated config.

Content addressing rather than a fixed filename because the failure it prevents is nasty and silent:
with a stable key, editing the script and re-booting a cluster can leave nodes fetching a cached or
half-replaced object, and you get an inconsistent fleet with no error anywhere. A digest in the key
makes "which script did this cluster actually run" answerable from the config alone, and makes the
upload idempotent — same bytes, same key, no-op.

**The nodes also need permission to read it.** The generated config must grant S3 read on that bucket
to both the head node and every queue, or node configuration fails at download with a 403:

```yaml
Iam:
  S3Access:
    - BucketName: <bucket>
      EnableWriteAccess: false
```

Omitting this is the single most likely reason a first live `boot` fails, so `generate_cluster_config`
must emit it unconditionally whenever a bootstrap URL is present.

---

## 7. `external.py`

```python
def require_tool(name: str) -> str:
    """Return the absolute path to an external tool.

    Raises:
        ToolMissingError: If the tool is not on PATH. Exits 5, with an install hint.
    """


def run(
    argv: Sequence[str], *, dry_run: bool, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess[str] | None:
    """Execute an external command, or print it when in dry-run.

    Args:
        argv: Command as an argument vector. Never a shell string.
        dry_run: When true, render the command and return None without executing.
        check: Raise ExternalCommandError (exit 6) on a non-zero return code.
        capture: Capture stdout/stderr rather than streaming to the terminal.

    Returns:
        The completed process, or None in dry-run.
    """
```

Two rules. **Always an argv list, never `shell=True`** — cluster names and paths come from the
environment, and a shell string turns any of them into an injection vector. **Rendering for display
must go through `shlex.join`**, so what dry-run prints is a command the user can actually paste,
correctly quoted. Hand-rolled `" ".join()` produces output that looks right and breaks on the first
path containing a space.

---

## 8. Generators

All three are pure functions: settings in, `str` out. No I/O, no environment access, no printing.
That is what makes them directly testable (§12) and it is what P3 requires.

### `generators/cluster_config.py`

```python
def build_cluster_config(settings: Settings, *, bootstrap_url: str | None) -> dict[str, Any]:
    """Build the ParallelCluster configuration as a plain dictionary."""


def render_cluster_config(settings: Settings, *, bootstrap_url: str | None) -> str:
    """Render the ParallelCluster configuration as YAML text."""
```

`render` is `yaml.safe_dump(build(...), sort_keys=False, default_flow_style=False, width=1000)`.
`sort_keys=False` preserves the intentional ordering below, which reads top-down as the cluster is
actually built. `width=1000` stops PyYAML from line-folding long S3 URLs into a form that is valid
YAML but unreadable in review — and review is the entire product of dry-run.

Emitted structure (ParallelCluster 3.x):

```yaml
Region: us-east-1
Image:
  Os: ubuntu2204
HeadNode:
  InstanceType: t3.medium
  Networking:
    SubnetId: subnet-aaaa
  Ssh:
    KeyName: my-keypair
  LocalStorage:
    RootVolume:
      Size: 50
  Iam:
    S3Access:
      - BucketName: my-bucket
        EnableWriteAccess: false
  CustomActions:
    OnNodeConfigured:
      Script: s3://my-bucket/hpcctl/bootstrap/install_engine_deps-1a2b3c4d.sh
Scheduling:
  Scheduler: slurm
  SlurmQueues:
    - Name: compute
      ComputeResources:
        - Name: compute-cr
          InstanceType: c5.large
          MinCount: 0
          MaxCount: 4
      Networking:
        SubnetIds:
          - subnet-aaaa
      Iam:
        S3Access:
          - BucketName: my-bucket
            EnableWriteAccess: false
      CustomActions:
        OnNodeConfigured:
          Script: s3://my-bucket/hpcctl/bootstrap/install_engine_deps-1a2b3c4d.sh
SharedStorage:
  - Name: shared
    StorageType: Ebs
    MountDir: /shared
    EbsSettings:
      Size: 50
      VolumeType: gp3
```

Three things worth defending. The `OnNodeConfigured` hook is attached to **both** the head node and
the queue, because the head node compiles nothing but must still be able to inspect and run the
engine. `SharedStorage` at `HPCCTL_SHARED_DIR` is not optional — it is what makes `deploy`'s target
visible to compute nodes (§5), and dropping it produces jobs that work on one node and fail on the
rest. `Os` defaults to `ubuntu2204` rather than `ubuntu2404` because it is supported across the
widest range of ParallelCluster 3.x releases; it is env-configurable, and the exact value should be
checked against the installed `pcluster` version on first live use.

### `generators/sbatch.py`

```python
def render_sbatch(settings: Settings, *, dag_remote_path: str, job_name: str) -> str:
    """Render the Slurm batch script for one DAG execution."""
```

```bash
#!/bin/bash
#SBATCH --job-name=bench-matmul-001
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#SBATCH --output=/shared/dags/bench-matmul-001-%j.out
#SBATCH --error=/shared/dags/bench-matmul-001-%j.err

set -euo pipefail

echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NUM_NODES} node(s), ${SLURM_NTASKS} task(s)"
srun --mpi=pmix /shared/engine/bin/engine --dag /shared/dags/bench-matmul-001.json
```

**Every `#SBATCH` directive must precede the first non-comment, non-blank line.** Slurm stops
scanning at the first real command, so a directive placed after `set -euo pipefail` is silently
ignored — the job runs with defaults instead of failing, which is far worse than an error. §12 makes
this a test rather than a convention. `--mpi=pmix` matches ParallelCluster's Slurm build; it is the
other half of the MPI risk flagged in §6.

### `generators/bootstrap.py`

```python
def bootstrap_path() -> Path:
    """Return the filesystem path to the packaged bootstrap script."""


def bootstrap_text() -> str:
    """Return the bootstrap script's contents."""


def bootstrap_digest() -> str:
    """Return the SHA-256 hex digest of the bootstrap script."""


def bootstrap_s3_url(settings: Settings) -> str:
    """Return the content-addressed S3 URL for the bootstrap script."""
```

---

## 9. Console UX, and a measured constraint on it

`console.py` owns a single `rich.console.Console`. Honour `--no-color` and the conventional
`NO_COLOR` environment variable.

**Rich truncates long lines, and a `Panel` around syntax makes it worse.** I measured this rather
than assuming it. Rendering a bash script containing a 224-character `apt-get install` line at
terminal width 80:

| Rendering | Full line preserved |
| --- | --- |
| `Syntax(text, "bash")` | **no** — truncated mid-token, no warning |
| `Syntax(text, "bash", word_wrap=True)` | yes |
| `Syntax(text, "bash")` with `crop=False` | **no** |
| `Panel(Syntax(text, "bash", word_wrap=True))` | **no** — the border steals width and re-truncates |
| plain `str` with `soft_wrap=True` | yes |

The `Panel` row is the trap, because panels are the obvious way to present a labelled artifact and
the failure is invisible: the output looks clean and is missing packages. Three consequences, all
binding:

1. **Artifact bodies are never wrapped in a `Panel`.** Use `rich.rule.Rule` for the label and print
   `Syntax(..., word_wrap=True)` at full width. Panels stay for short content — the env-var
   placeholder summary, the confirmation blurb — where truncation cannot occur.
2. **`--raw` writes artifacts with `sys.stdout.write`**, bypassing rich entirely, so
   `hpcctl boot --raw | ...` yields exact bytes. This is the supported path for piping into
   `bash -n` or `yaml.safe_load`.
3. **Tests never assert on rendered output** (§12). They call the generator and check the returned
   string.

```python
def render_artifact(title: str, body: str, lexer: str) -> None:
    """Print a generated artifact with a labelled rule and syntax highlighting.

    Uses a Rule rather than a Panel, and word_wrap, because Panel borders reduce the
    available width and silently truncate long lines.

    Args:
        title: Human-readable artifact label, e.g. "cluster config (YAML)".
        body: Exact artifact text.
        lexer: Pygments lexer name, one of "yaml", "bash", or "json".
    """


def render_placeholder_warning(settings: Settings) -> None:
    """Warn that required configuration was replaced with placeholders."""


def render_command(argv: Sequence[str]) -> None:
    """Print an external command as a copy-pasteable, shell-quoted line."""
```

Also set `Console(stderr=True)` for a second console used by warnings and errors, so stdout stays
clean enough to pipe.

---

## 10. Artifact paths and the `.gitignore` interaction

The repo-root `.gitignore` ignores `*.json`, `*.yaml`, and `*.yml` globally, with narrow negations
for `/shared/**`, `/.github/**`, `*.example.*`, and `.pre-commit-config.yaml`. It also already
ignores `*.sbatch.generated` and `slurm-*.out`. Consequences:

- **No committed YAML template**, as instructed. The config is built as a `dict` and rendered with
  `yaml.safe_dump`. There is no `.yaml` or `.yaml.j2` file anywhere in `hpcctl`. Discoverability is
  served by `--dry-run` output plus the committed `.env.example`, which is the one config artifact
  with an explicit negation.
- **Generated artifacts land in `HPCCTL_RUN_DIR`** (default `./.hpcctl-run/`):
  - `<cluster>-config.yaml` — already ignored by the global `*.yaml` rule, in any directory.
  - `<job>.sbatch.generated` — already ignored by the existing `*.sbatch.generated` rule.
  - The run directory is created with `parents=True, exist_ok=True` on first use.
- **The bootstrap `.sh` is committed** — no `.gitignore` rule touches `*.sh`.

> **One-line change I recommend but did not make.** `.hpcctl-run/` is not itself ignored. Both
> artifact types are ignored by extension today, so nothing leaks, but that safety is incidental: the
> moment someone writes a `.txt` log or a `.sh` into the run directory it becomes committable. Adding
> `.hpcctl-run/` to the "HPC run artifacts" section makes the guarantee structural rather than
> coincidental. I left `.gitignore` alone because it is shared and Reviewer has been editing it.

Also note for whoever owns secrets hygiene: `deploy` and `submit` read `HPCCTL_SSH_KEY_PATH` but must
never print the key's *contents* — only its path — and `*.pem` and `id_rsa*` are already ignored, so
a key accidentally dropped in the repo will not commit.

---

## 11. DAG validation

```python
def load_schema(path: Path | None = None) -> dict[str, Any]:
    """Load the DAG JSON Schema from /shared/dag_schema.json."""


def validate_dag_file(path: Path, *, schema_path: Path | None = None) -> dict[str, Any]:
    """Parse and validate a serialized DAG document.

    Returns:
        The parsed document, when valid.

    Raises:
        DagValidationError: If the file is not valid JSON, or does not conform to the schema.
            Carries every schema error, not just the first.
    """
```

Use `jsonschema.Draft202012Validator` explicitly rather than `validate()`'s auto-detection, and call
`check_schema` once on load so a corrupted contract is reported as a contract bug rather than as a
DAG bug. Collect errors with `sorted(v.iter_errors(doc), key=str)` — stable order, all problems at
once. Render as a table of `/`-joined `absolute_path` and `message`.

Additionally warn (do not fail) when `metadata.schema_version`'s major component differs from the
major version of the loaded schema's own documented version. Schema 1.1.0's compatibility rule is
that major mismatches are incompatible, and catching that locally is much cheaper than catching it
after a job launches. It is a warning rather than an error because the schema is the authority: if
the document validates structurally, it is loadable.

---

## 12. Test checklist

The whole point of the dry-run architecture is that all of this is testable today, with no AWS
account. `typer.testing.CliRunner` is available and its `invoke()` accepts an `env` mapping, which is
exactly what the env-var matrix needs; `Result.stderr` is separate from `stdout`. Verified working
against the existing `app`.

- **`test_config.py`** — every default in §4; `AWS_REGION` falling back to `AWS_DEFAULT_REGION`;
  `HPCCTL_COMPUTE_SUBNET_ID` falling back to the head subnet; dry-run producing
  `<<<UNSET:HPCCTL_KEY_NAME>>>` and listing it in `missing`; `live=True` raising `MissingConfigError`
  and reporting **all** missing vars at once, not just the first; `strict=True` failing in dry-run.
- **`test_cluster_config.py`** — `yaml.safe_load(render_cluster_config(...))` parses; asserted keys
  present (`Region`, `Image.Os`, `HeadNode.Ssh.KeyName`, `Scheduling.Scheduler == "slurm"`); the
  `OnNodeConfigured.Script` on head node **and** queue equal the content-addressed URL; `Iam.S3Access`
  present in both places; `SharedStorage[0].MountDir == HPCCTL_SHARED_DIR`; a config full of
  placeholders still parses as YAML.
- **`test_sbatch.py`** — `bash -n` on the rendered script (via `subprocess`, `bash` is present);
  every `#SBATCH` directive appears before the first non-comment non-blank line — assert this
  positionally, it is the silent-failure case from §8; `--nodes`/`--ntasks`/`--time` reflect
  overrides; the DAG path and engine binary are the resolved remote paths.
- **`test_bootstrap.py`** — `bash -n` on the packaged script; it contains `set -euo pipefail`,
  `DEBIAN_FRONTEND=noninteractive`, and `--force-confold`; it contains no bare `apt install` or
  `apt-get upgrade` (regex assertions — these are the rules most likely to erode); `bootstrap_path()`
  resolves via `importlib.resources`; `bootstrap_digest()` is stable and the S3 URL embeds its first
  8 characters. Add `shellcheck` to CI if available — it is not installed on this VM, so keep it
  optional and never let its absence fail the suite.
- **`test_validation.py`** — a valid DAG passes; a DAG with a bad `op` fails with a path-bearing
  message; malformed JSON exits 4 rather than raising; multiple schema errors are all reported. Lift
  fixtures from the 58-case harness at `/tmp/check_dag_schema.py`.
- **`test_cli.py`** — for each of `boot`, `deploy`, `submit`, `status`, `destroy`: dry-run exits 0
  with **no** env vars set (this is P1, and it is the single most important test in the suite);
  `--execute` with no tools installed exits 5, not 1; `destroy` with a wrong typed name exits 7;
  `destroy` with no TTY and no `--yes` exits 7 without hanging; `submit --validate-only` on a bad DAG
  exits 4; `HPCCTL_DRY_RUN=1` defeats `--execute`.
- **`test_raw_output.py`** — `--raw` output of the bootstrap script passes `bash -n`, and `--raw`
  output of the config passes `yaml.safe_load`. Then assert the negative case that motivates §9: the
  *rendered* form of a deliberately long line is **not** byte-identical to the source, so nobody
  later "simplifies" the raw path away.

No test should require network, credentials, `pcluster`, or `aws`. Any test that would is a design
bug in the thing being tested.

---

## 13. Decisions for the Reviewer

1. **Dry-run is the default; `--execute` opts in** (P2). The safest default given no credits, and
   both flags exist so CLAUDE.md §2 is satisfied regardless.
2. **Missing env vars are placeholders in dry-run, fatal when live** (§4). Without this, dry-run
   could not run at all today, which would make the entire CLI untestable.
3. **`hpcctl` does not depend on `tasks`** (P4). Validation goes through `/shared/dag_schema.json`,
   the same contract the engine reads.
4. **Artifact bodies are never rendered inside a rich `Panel`** (§9), because measurement shows the
   border causes silent truncation. This partially contradicts the requested "panels" UX; panels are
   used for short content only.
5. **The bootstrap script is content-addressed in S3** (§6) rather than uploaded under a stable key,
   to make a cluster's provisioning script identifiable from its config and to keep re-uploads
   idempotent.
6. **`destroy` requires typing the cluster name**, not `y/N`, and never prompts in dry-run (§5).
7. **Open risk, unresolvable without an account:** Ubuntu-archive `libopenmpi-dev` may conflict with
   the ParallelCluster AMI's own MPI stack (§6). Flagged with a concrete first-boot check rather than
   guessed at.
