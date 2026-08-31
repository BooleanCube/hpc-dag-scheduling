# hpcctl — cluster control CLI

Typer CLI managing the AWS ParallelCluster lifecycle for the
[HPC DAG Scheduler Research Baseline](../README.md): create the cluster, ship the engine,
submit DAGs, download results, tear down.

Commands **execute by default**; every one takes `--dry-run` to print exactly what it would do
— for free, offline, with nothing configured. Three safety rails: missing config fails before
the first AWS call, `--dry-run` previews anything, and exporting `HPCCTL_DRY_RUN=1` forces
dry-run globally (set it whenever you're not deliberately spending).

## The lifecycle

```
        boot ──► deploy ──► submit ──► result ──► destroy
       (~15m)     (secs)     (secs)     (secs)     (~4m)
      $ starts                                   $ stops
```

Everything runs from the repository root (or any subdirectory — the repo is a uv
workspace, and hpcctl locates the engine and the DAG contract from the repo itself, never
from your working directory):

```bash
set -a && . hpcctl/.env && set +a              # once per shell 

uv run hpcctl boot                             # 1. create the cluster
uv run hpcctl deploy                           # 2. build the engine, ship the binary
uv run hpcctl submit cases/wide_skewed_001.py -n 4
                                               # 3. compile + run a case; prints a job ID
uv run hpcctl result 2                         # 4. download that job's outputs
uv run hpcctl destroy                          # 5. stop paying (type the name to confirm)
```

No IP bookkeeping: remote commands discover the head node's current address from the cluster
every time, so a rebooted cluster (fresh IP) just works. The first job after `boot` waits
~5 minutes for compute nodes to cold-start; warm jobs run in seconds. Poll anything with
`uv run hpcctl status` (`--no-queue` to skip SSH).

**Submitting:** `submit` takes a Python case script (it runs the script with an output
path as `argv[1]` — the `vmath.emit` convention — then validates and ships the document)
or an already-serialized DAG JSON. The benchmark suite lives in
[`/cases`](../cases/README.md).

**Job geometry:** `-n N` = N compute instances running **N+1 MPI ranks** — cyclic placement
gives every instance one worker and wraps the scheduler (the highest rank, by engine
convention) onto the first instance beside worker rank 0, so no machine idles under the
scheduler. Omit `-n` for `HPCCTL_NODES` (default 1 → 2 ranks); the count must sit within the
queue's `HPCCTL_MIN_NODES`/`HPCCTL_MAX_NODES` bounds or submit refuses it locally. `--ntasks`
overrides the derivation for irregular experiments.

**Results:** every job runs with cwd = `/shared/dags/results/<job id>/` on the shared
filesystem. `result <job id>` downloads the Slurm stdout log, the stderr log (absent for a
healthy quiet job), and every file the engine wrote, into local `./results/<job id>/`
(git-ignored). Healthy stdout ends with `ENGINE_OK ranks=<N> …` with ranks across multiple
hostnames; a rank-0-only run means an MPI stack mismatch.

## One-time AWS setup

Everything only you can do, in order. (All done on this machine already — kept for a fresh box.)

1. **AWS CLI v2 + credentials:**

   ```bash
   curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
   unzip -q /tmp/awscliv2.zip -d /tmp && sudo /tmp/aws/install
   aws configure                  # keys for an IAM user; AdministratorAccess is the
   aws sts get-caller-identity    # pragmatic choice for a personal research account
   ```

2. **An ed25519 EC2 key pair** (Ubuntu 24.04 AMIs reject RSA, and AWS creates RSA by default):

   ```bash
   aws ec2 create-key-pair --key-name hpc-dag-key-ed25519 --key-type ed25519 \
     --query KeyMaterial --output text > ~/.ssh/hpc-dag-key-ed25519.pem
   chmod 400 ~/.ssh/hpc-dag-key-ed25519.pem
   ```

3. **A subnet with public IPs** (any default-VPC subnet):

   ```bash
   aws ec2 describe-subnets \
     --query 'Subnets[?MapPublicIpOnLaunch].[SubnetId,AvailabilityZone]' --output table
   ```

4. **An S3 bucket** for the bootstrap script: `aws s3 mb s3://<globally-unique-name>`

5. **The ParallelCluster CLI**, inside the workspace venv (from the repo root):
   `uv pip install aws-parallelcluster`

6. **Engine build tooling** (see [`engine/README.md`](../engine/README.md)) — `deploy`
   compiles the engine automatically before every sync (`--no-build` to skip), but the
   compiler and MPI it needs come from the same bootstrap script cluster nodes run:

   ```bash
   bash hpcctl/src/hpcctl/bootstrap/install_engine_deps.sh
   ```

## Configuration

`cp hpcctl/.env.example hpcctl/.env`, fill in three values, then `set -a && . hpcctl/.env && set +a`. hpcctl
never auto-loads `.env` — a tool that spends money should not read hidden config. The
committed [`.env.example`](.env.example) documents every variable; the ones you must set:

| Variable | What |
| -------- | ---- |
| `HPCCTL_KEY_NAME` | The EC2 key pair *name* from step 2 (and `HPCCTL_SSH_KEY_PATH` to its `.pem`) |
| `HPCCTL_HEAD_SUBNET_ID` | The subnet from step 3 |
| `HPCCTL_BOOTSTRAP_BUCKET` | The bucket from step 4 |

Leave `HPCCTL_HEAD_NODE_HOST` unset (auto-discovered; setting it pins the address). Keep
`HPCCTL_OS` matched to the OS you build the engine on — a newer-glibc binary won't run on an
older cluster. Everything else has sensible defaults: instance types (`t3.medium` head,
`c5.large` compute), fleet limits (`MIN_NODES=0` so idle compute self-terminates), the 50 GB
`/shared` volume, and paths.

### Cost (us-east-1, on-demand, defaults)

Head node ~$0.04/hr + EBS ~$0.01/hr while the cluster exists; compute ~$0.085/hr per c5.large
**only while jobs run**. A full boot → run → destroy session costs well under $1; an idle
forgotten cluster ~$1.30/day. End every session with `hpcctl destroy` — recreating takes
~15 minutes and is deterministic from `.env`.

## Design

How the module works, and the decisions that keep it safe to operate:

- **Layout.** `commands/` (one module per command) → `generators/` (pure functions:
  settings in, artifact text out — cluster YAML, sbatch scripts, the packaged
  `bootstrap/install_engine_deps.sh`) → `external.py` (subprocess wrapper; always argv lists,
  never `shell=True`; dry-run prints via `shlex.join` so output is paste-able) →
  `config.py` (sole reader of the environment) → `discovery.py` (head-node resolution).
- **Dry-run runs fully offline.** No credentials, tools, network, or env vars needed: unset
  required values become `<<<UNSET:NAME>>>` placeholders (valid YAML, rejected by every AWS
  API, impossible to mistake for real). Live runs fail on missing config before the first
  API call, listing every missing variable at once. This is what makes the whole CLI testable
  without an account — the suite never touches AWS and actively hides real `pcluster`/`aws`
  installs from itself.
- **hpcctl validates the contract, not the builder.** `submit` checks DAGs against
  `/shared/dag_schema.json` with `jsonschema` — exactly what the engine reads — and never
  imports `/vmath`. Case scripts are compiled in a **subprocess** for the same reason;
  it is the one external command a dry-run runs (local, free, and required for the
  preview), and never an AWS-touching tool.
- **The bootstrap script is content-addressed in S3** (SHA-256 in the key), so which script a
  cluster ran is answerable from its config, and re-uploads are idempotent. The same script
  provisions cluster nodes and dev boxes.
- **Slurm tools are invoked by absolute path** (`HPCCTL_SLURM_BIN`, default `/opt/slurm/bin`):
  non-interactive SSH shells and sbatch-propagated environments do not have Slurm on PATH —
  both bit on the first live run.
- **SSH policy:** `StrictHostKeyChecking=accept-new` (pins after first contact), never `no`;
  key *contents* are never read or printed.
- **Byte-exact output on demand:** `--raw` bypasses the rich renderer (which reflows long
  lines) so `boot --raw | …` yields exact artifact bytes.
- **Exit codes are a stable contract:** 0 ok · 1 internal bug · 2 usage · 3 config ·
  4 invalid DAG · 5 missing tool · 6 external command failed · 7 aborted confirmation ·
  8 cluster absent/failed.

## Development

```bash
cd hpcctl   # tool configs live in the member project; the venv is the workspace root's
uv sync && uv run pytest && uv run ruff check . && uv run ruff format . && uv run mypy .
```
