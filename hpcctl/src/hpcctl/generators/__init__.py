"""Pure artifact generators: settings in, ``str`` out.

No I/O, no environment access, no printing. That is what makes them directly testable and what
keeps rendering a separate, lossy presentation step (see :mod:`hpcctl.console`).
"""

from hpcctl.generators.bootstrap import (
    bootstrap_digest,
    bootstrap_path,
    bootstrap_s3_url,
    bootstrap_text,
    upload_argv,
)
from hpcctl.generators.cluster_config import (
    build_cluster_config,
    create_cluster_argv,
    render_cluster_config,
)
from hpcctl.generators.sbatch import (
    remote_dag_path,
    remote_sbatch_path,
    render_sbatch,
    sbatch_directives,
)

__all__ = [
    "bootstrap_digest",
    "bootstrap_path",
    "bootstrap_s3_url",
    "bootstrap_text",
    "build_cluster_config",
    "create_cluster_argv",
    "remote_dag_path",
    "remote_sbatch_path",
    "render_cluster_config",
    "render_sbatch",
    "sbatch_directives",
    "upload_argv",
]
