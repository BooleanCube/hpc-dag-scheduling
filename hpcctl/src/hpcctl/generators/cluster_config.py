"""Build the AWS ParallelCluster configuration.

The config is assembled as a plain ``dict`` and rendered with ``yaml.safe_dump``. There is no
committed YAML template anywhere in ``hpcctl``: the repo-root ``.gitignore`` ignores ``*.yaml``
globally, and discoverability is served by dry-run output plus the committed ``.env.example``.

Three structural choices are load-bearing rather than cosmetic:

* ``CustomActions.OnNodeConfigured`` is attached to **both** the head node and every queue. The
  head node compiles nothing, but it must still be able to inspect and run the engine.
* ``Iam.S3Access`` is emitted in both places whenever a bootstrap URL is present. Omitting it is
  the single most likely reason a first live ``boot`` fails: node configuration dies at download
  with a 403.
* ``SharedStorage`` at ``HPCCTL_SHARED_DIR`` is not optional. It is what makes ``deploy``'s target
  visible to compute nodes; dropping it produces jobs that work on one node and fail on the rest.
"""

from typing import Any

import yaml

from hpcctl.config import Settings


def _s3_access(bucket: str) -> list[dict[str, Any]]:
    """Build the ``Iam.S3Access`` block granting read on the bootstrap bucket.

    Args:
        bucket: Bucket name to grant read access to.

    Returns:
        A single-entry S3 access list, read-only.
    """
    return [{"BucketName": bucket, "EnableWriteAccess": False}]


def build_cluster_config(settings: Settings, *, bootstrap_url: str | None) -> dict[str, Any]:
    """Build the ParallelCluster configuration as a plain dictionary.

    Args:
        settings: Resolved settings supplying every environment-specific value.
        bootstrap_url: Content-addressed ``s3://`` URL of the bootstrap script, or ``None`` to
            omit the node-configuration hook and its IAM grant entirely.

    Returns:
        A ParallelCluster 3.x configuration document, ordered so it reads top-down the way the
        cluster is actually built.
    """
    head_node: dict[str, Any] = {
        "InstanceType": settings.head_instance_type,
        "Networking": {"SubnetId": settings.head_subnet_id},
        "Ssh": {"KeyName": settings.key_name},
        "LocalStorage": {"RootVolume": {"Size": settings.shared_volume_gb}},
    }
    queue: dict[str, Any] = {
        "Name": settings.queue_name,
        "ComputeResources": [
            {
                "Name": f"{settings.queue_name}-cr",
                "InstanceType": settings.compute_instance_type,
                "MinCount": settings.min_nodes,
                "MaxCount": settings.max_nodes,
            }
        ],
        "Networking": {"SubnetIds": [settings.compute_subnet_id]},
    }

    if bootstrap_url is not None:
        # Both the grant and the hook go on the head node AND the queue; a queue without the
        # grant fails node configuration with a 403 at download time.
        access = _s3_access(settings.bootstrap_bucket)
        hook = {"OnNodeConfigured": {"Script": bootstrap_url}}
        head_node["Iam"] = {"S3Access": access}
        head_node["CustomActions"] = hook
        queue["Iam"] = {"S3Access": _s3_access(settings.bootstrap_bucket)}
        queue["CustomActions"] = {"OnNodeConfigured": {"Script": bootstrap_url}}

    return {
        "Region": settings.region,
        "Image": {"Os": settings.os_image},
        "HeadNode": head_node,
        "Scheduling": {"Scheduler": "slurm", "SlurmQueues": [queue]},
        "SharedStorage": [
            {
                "Name": "shared",
                "StorageType": "Ebs",
                "MountDir": settings.shared_dir,
                "EbsSettings": {"Size": settings.shared_volume_gb, "VolumeType": "gp3"},
            }
        ],
    }


def render_cluster_config(settings: Settings, *, bootstrap_url: str | None) -> str:
    """Render the ParallelCluster configuration as YAML text.

    ``sort_keys=False`` preserves the intentional top-down ordering. ``width=1000`` stops PyYAML
    from line-folding long S3 URLs into a form that is valid YAML but unreadable in review -- and
    review is the entire product of dry-run.

    Args:
        settings: Resolved settings supplying every environment-specific value.
        bootstrap_url: Content-addressed ``s3://`` URL of the bootstrap script, or ``None``.

    Returns:
        The configuration as a YAML document.
    """
    document = build_cluster_config(settings, bootstrap_url=bootstrap_url)
    rendered: str = yaml.safe_dump(
        document,
        sort_keys=False,
        default_flow_style=False,
        width=1000,
    )
    return rendered


def create_cluster_argv(settings: Settings, config_path: str) -> list[str]:
    """Build the ``pcluster create-cluster`` command.

    Args:
        settings: Resolved settings supplying the cluster name and region.
        config_path: Path to the rendered configuration file.

    Returns:
        The ``pcluster`` argument vector.
    """
    return [
        "pcluster",
        "create-cluster",
        "--cluster-name",
        settings.cluster_name,
        "--cluster-configuration",
        config_path,
        "--region",
        settings.region,
    ]
