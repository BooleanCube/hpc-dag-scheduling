"""The ``boot`` command: create the ParallelCluster cluster."""

from hpcctl import console
from hpcctl.commands.options import (
    DryRunOption,
    EmitDirOption,
    RawOption,
    StrictOption,
    artifact_dir,
    resolve_dry_run,
)
from hpcctl.config import REQUIRED_ALWAYS, load_settings
from hpcctl.external import require_tools, run
from hpcctl.generators.bootstrap import (
    bootstrap_digest,
    bootstrap_s3_url,
    bootstrap_text,
    upload_argv,
)
from hpcctl.generators.cluster_config import create_cluster_argv, render_cluster_config


def boot(
    dry_run: DryRunOption = True,
    strict: StrictOption = False,
    emit_dir: EmitDirOption = None,
    raw: RawOption = False,
) -> None:
    """Create the ParallelCluster cluster.

    In dry-run this prints the three artifacts that would be used -- the Ubuntu bootstrap script
    that nodes will execute, the generated ParallelCluster YAML, and the exact
    ``pcluster create-cluster`` invocation -- and writes them to the run directory.
    """
    dry_run = resolve_dry_run(dry_run)
    settings = load_settings(live=not dry_run, strict=strict, required=REQUIRED_ALWAYS)

    script = bootstrap_text()
    digest = bootstrap_digest()
    url = bootstrap_s3_url(settings)
    upload = upload_argv(settings)
    config_text = render_cluster_config(settings, bootstrap_url=url)

    target = artifact_dir(settings, emit_dir)
    config_path = target / f"{settings.cluster_name}-config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    script_copy = target / f"install_engine_deps-{digest[:8]}.sh"
    script_copy.write_text(script, encoding="utf-8")

    create = create_cluster_argv(settings, str(config_path))

    if dry_run:
        _print_plan(script, config_text, upload, create, digest=digest, url=url, raw=raw)
        if not raw:
            console.render_notice(f"artifacts written to {target}")
            console.render_placeholder_warning(settings)
        return

    # Order matters: each step is a precondition for the next.
    require_tools("aws", "pcluster")
    console.render_notice(f"publishing bootstrap script to {url}")
    run(upload, dry_run=False)
    console.render_notice(f"creating cluster {settings.cluster_name!r} in {settings.region}")
    completed = run(create, dry_run=False)
    if completed is not None and completed.stdout:
        console.out().print(completed.stdout.strip())
    console.render_notice(
        "cluster creation is asynchronous; poll it with 'hpcctl status --execute'"
    )


def _print_plan(
    script: str,
    config_text: str,
    upload: list[str],
    create: list[str],
    *,
    digest: str,
    url: str,
    raw: bool,
) -> None:
    """Print the three dry-run artifacts in dependency order.

    Dependency order rather than alphabetical so the output reads as a narrative: the script is
    hashed, the hash names the S3 object, the config points at that object, and the create
    command points at the config.

    Args:
        script: Exact bootstrap script text.
        config_text: Rendered cluster configuration YAML.
        upload: The ``aws s3 cp`` argument vector.
        create: The ``pcluster create-cluster`` argument vector.
        digest: Full SHA-256 of the bootstrap script.
        url: Content-addressed S3 URL the config will reference.
        raw: Bypass rich rendering entirely and emit exact bytes.
    """
    if raw:
        console.write_raw("bootstrap", script)
        console.write_raw("cluster-config", config_text)
        console.write_raw(
            "commands", "\n".join([console.format_command(upload), console.format_command(create)])
        )
        return

    console.render_artifact("1/3 bootstrap script (bash)", script, "bash")
    console.render_notice(f"sha256 {digest}")
    console.render_notice(f"would upload to {url}")
    console.render_command(upload)

    console.render_artifact("2/3 cluster configuration (YAML)", config_text, "yaml")

    console.render_artifact(
        "3/3 pcluster invocation (bash)", console.format_command(create), "bash"
    )
