"""Tests for the generated ParallelCluster configuration.

Assertions go against ``yaml.safe_load`` of the generator's return value, never against CLI
stdout: the console layer is lossy by design (see :mod:`hpcctl.console`), so scraping it would
make these tests measure the renderer instead of the config.
"""

from typing import Any

import pytest
import yaml

from hpcctl.config import Settings, load_settings
from hpcctl.generators.bootstrap import bootstrap_s3_url
from hpcctl.generators.cluster_config import (
    build_cluster_config,
    create_cluster_argv,
    render_cluster_config,
)

BOOTSTRAP_URL = "s3://my-bucket/hpcctl/bootstrap/install_engine_deps-1a2b3c4d.sh"


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Return settings with every required variable supplied.

    Args:
        monkeypatch: pytest's environment patcher.

    Returns:
        Fully specified settings, so assertions read against real-looking values.
    """
    for name, value in {
        "AWS_REGION": "us-east-1",
        "HPCCTL_KEY_NAME": "my-keypair",
        "HPCCTL_HEAD_SUBNET_ID": "subnet-aaaa",
        "HPCCTL_BOOTSTRAP_BUCKET": "my-bucket",
        "HPCCTL_HEAD_NODE_HOST": "1.2.3.4",
    }.items():
        monkeypatch.setenv(name, value)
    return load_settings(live=False)


def parsed(settings: Settings, *, url: str | None = BOOTSTRAP_URL) -> dict[str, Any]:
    """Render the config and parse it back as YAML.

    Args:
        settings: Settings to render from.
        url: Bootstrap URL to embed, or ``None`` to omit the hook.

    Returns:
        The parsed configuration document.
    """
    loaded: dict[str, Any] = yaml.safe_load(render_cluster_config(settings, bootstrap_url=url))
    return loaded


class TestParsesAsYaml:
    def test_render_output_parses(self, configured: Settings) -> None:
        assert isinstance(parsed(configured), dict)

    def test_placeholder_config_still_parses(self, settings: Settings) -> None:
        """The placeholder format is deliberately a legal YAML scalar."""
        document = parsed(settings, url=bootstrap_s3_url(settings))
        assert document["Region"] == "<<<UNSET:AWS_REGION>>>"

    def test_no_yaml_template_is_committed(self) -> None:
        """The root .gitignore ignores *.yaml, so the config must be built as a dict."""
        from pathlib import Path

        package = Path(__file__).resolve().parents[1] / "src" / "hpcctl"
        assert list(package.rglob("*.yaml")) == []
        assert list(package.rglob("*.yml")) == []
        assert list(package.rglob("*.j2")) == []


class TestStructuralAssertions:
    """The nine keys the engine and ParallelCluster both depend on."""

    def test_1_region(self, configured: Settings) -> None:
        assert parsed(configured)["Region"] == "us-east-1"

    def test_2_image_os(self, configured: Settings) -> None:
        assert parsed(configured)["Image"]["Os"] == "ubuntu2204"

    def test_3_head_node_key_name(self, configured: Settings) -> None:
        assert parsed(configured)["HeadNode"]["Ssh"]["KeyName"] == "my-keypair"

    def test_4_scheduler_is_slurm(self, configured: Settings) -> None:
        assert parsed(configured)["Scheduling"]["Scheduler"] == "slurm"

    def test_5_head_node_on_node_configured(self, configured: Settings) -> None:
        head = parsed(configured)["HeadNode"]
        assert head["CustomActions"]["OnNodeConfigured"]["Script"] == BOOTSTRAP_URL

    def test_6_queue_on_node_configured(self, configured: Settings) -> None:
        queue = parsed(configured)["Scheduling"]["SlurmQueues"][0]
        assert queue["CustomActions"]["OnNodeConfigured"]["Script"] == BOOTSTRAP_URL

    def test_7_head_node_s3_access(self, configured: Settings) -> None:
        access = parsed(configured)["HeadNode"]["Iam"]["S3Access"]
        assert access == [{"BucketName": "my-bucket", "EnableWriteAccess": False}]

    def test_8_queue_s3_access(self, configured: Settings) -> None:
        """Omitting this is the most likely reason a first live boot fails, with a 403."""
        queue = parsed(configured)["Scheduling"]["SlurmQueues"][0]
        assert queue["Iam"]["S3Access"] == [{"BucketName": "my-bucket", "EnableWriteAccess": False}]

    def test_9_shared_storage_mount_dir(self, configured: Settings) -> None:
        assert parsed(configured)["SharedStorage"][0]["MountDir"] == "/shared"

    def test_hook_is_on_both_head_and_queue(self, configured: Settings) -> None:
        """The head node compiles nothing but must still run and inspect the engine."""
        document = parsed(configured)
        head = document["HeadNode"]["CustomActions"]["OnNodeConfigured"]["Script"]
        queue = document["Scheduling"]["SlurmQueues"][0]["CustomActions"]["OnNodeConfigured"][
            "Script"
        ]
        assert head == queue == BOOTSTRAP_URL


class TestEnvironmentIsHonoured:
    def test_shared_dir_drives_the_mount_point(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_SHARED_DIR", "/mnt/fsx")
        assert parsed(load_settings(live=False))["SharedStorage"][0]["MountDir"] == "/mnt/fsx"

    def test_queue_name_and_compute_resource_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_QUEUE_NAME", "gpu")
        queue = parsed(load_settings(live=False))["Scheduling"]["SlurmQueues"][0]
        assert queue["Name"] == "gpu"
        assert queue["ComputeResources"][0]["Name"] == "gpu-cr"

    def test_node_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_MIN_NODES", "2")
        monkeypatch.setenv("HPCCTL_MAX_NODES", "32")
        resource = parsed(load_settings(live=False))["Scheduling"]["SlurmQueues"][0][
            "ComputeResources"
        ][0]
        assert resource["MinCount"] == 2
        assert resource["MaxCount"] == 32

    def test_instance_types(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_HEAD_INSTANCE_TYPE", "m5.large")
        monkeypatch.setenv("HPCCTL_COMPUTE_INSTANCE_TYPE", "c6i.8xlarge")
        document = parsed(load_settings(live=False))
        assert document["HeadNode"]["InstanceType"] == "m5.large"
        resource = document["Scheduling"]["SlurmQueues"][0]["ComputeResources"][0]
        assert resource["InstanceType"] == "c6i.8xlarge"

    def test_compute_subnet_is_used_for_the_queue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_HEAD_SUBNET_ID", "subnet-head")
        monkeypatch.setenv("HPCCTL_COMPUTE_SUBNET_ID", "subnet-compute")
        document = parsed(load_settings(live=False))
        assert document["HeadNode"]["Networking"]["SubnetId"] == "subnet-head"
        queue = document["Scheduling"]["SlurmQueues"][0]
        assert queue["Networking"]["SubnetIds"] == ["subnet-compute"]

    def test_volume_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_SHARED_VOLUME_GB", "200")
        document = parsed(load_settings(live=False))
        assert document["SharedStorage"][0]["EbsSettings"]["Size"] == 200

    def test_os_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_OS", "ubuntu2404")
        assert parsed(load_settings(live=False))["Image"]["Os"] == "ubuntu2404"


class TestWithoutBootstrap:
    def test_hook_and_grant_are_both_omitted(self, configured: Settings) -> None:
        document = parsed(configured, url=None)
        assert "CustomActions" not in document["HeadNode"]
        assert "Iam" not in document["HeadNode"]
        assert "CustomActions" not in document["Scheduling"]["SlurmQueues"][0]

    def test_config_is_still_structurally_complete(self, configured: Settings) -> None:
        document = parsed(configured, url=None)
        assert document["Scheduling"]["Scheduler"] == "slurm"
        assert document["SharedStorage"][0]["MountDir"] == "/shared"


class TestRenderingOptions:
    def test_key_order_is_preserved(self, configured: Settings) -> None:
        """sort_keys=False keeps the document readable top-down as the cluster is built."""
        text = render_cluster_config(configured, bootstrap_url=BOOTSTRAP_URL)
        keys = [line.split(":")[0] for line in text.splitlines() if line and line[0].isalpha()]
        assert keys == ["Region", "Image", "HeadNode", "Scheduling", "SharedStorage"]

    def test_long_urls_are_not_line_folded(self, configured: Settings) -> None:
        """width=1000 keeps S3 URLs on one line; review is the entire product of dry-run."""
        long_url = "s3://bucket/" + "a" * 200 + "/install_engine_deps-1a2b3c4d.sh"
        text = render_cluster_config(configured, bootstrap_url=long_url)
        assert long_url in text

    def test_no_flow_style(self, configured: Settings) -> None:
        text = render_cluster_config(configured, bootstrap_url=BOOTSTRAP_URL)
        assert "{" not in text

    def test_build_returns_a_plain_dict(self, configured: Settings) -> None:
        document = build_cluster_config(configured, bootstrap_url=BOOTSTRAP_URL)
        assert type(document) is dict


class TestCreateClusterArgv:
    def test_includes_name_config_and_region(self, configured: Settings) -> None:
        argv = create_cluster_argv(configured, "/tmp/cfg.yaml")
        assert argv[:2] == ["pcluster", "create-cluster"]
        assert "--cluster-name" in argv
        assert argv[argv.index("--cluster-configuration") + 1] == "/tmp/cfg.yaml"
        assert argv[argv.index("--region") + 1] == "us-east-1"

    def test_is_an_argv_list_not_a_shell_string(self, configured: Settings) -> None:
        """Never shell=True: cluster names come from the environment."""
        argv = create_cluster_argv(configured, "/tmp/cfg.yaml")
        assert all(isinstance(part, str) for part in argv)
        assert not any(" " in part for part in argv[:2])
