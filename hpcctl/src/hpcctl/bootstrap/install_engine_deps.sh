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
