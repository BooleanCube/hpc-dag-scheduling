"""Tests for the environment-variable contract.

The asymmetry between dry-run and live resolution is the whole reason this CLI is testable
without an AWS account, so it gets the densest coverage here.
"""

import dataclasses
from pathlib import Path

import pytest

from hpcctl.config import (
    REQUIRED_ALWAYS,
    REQUIRED_FOR_REMOTE,
    Settings,
    color_disabled,
    dry_run_forced,
    is_placeholder,
    load_settings,
    placeholder,
)
from hpcctl.errors import InvalidConfigError, MissingConfigError

FULL_ENV: dict[str, str] = {
    "AWS_REGION": "us-east-1",
    "HPCCTL_KEY_NAME": "kp",
    "HPCCTL_HEAD_SUBNET_ID": "subnet-aaaa",
    "HPCCTL_BOOTSTRAP_BUCKET": "bucket",
    "HPCCTL_HEAD_NODE_HOST": "1.2.3.4",
}
"""Every variable that has no default, so ``missing`` comes back empty."""


class TestDefaults:
    def test_every_documented_default(self) -> None:
        s = load_settings(live=False)
        assert s.cluster_name == "hpc-dag-baseline"
        assert s.os_image == "ubuntu2204"
        assert s.head_instance_type == "t3.medium"
        assert s.compute_instance_type == "c5.large"
        assert s.queue_name == "compute"
        assert s.min_nodes == 0
        assert s.max_nodes == 4
        assert s.shared_dir == "/shared"
        assert s.shared_volume_gb == 50
        assert s.bootstrap_prefix == "hpcctl/bootstrap"
        assert s.ssh_user == "ubuntu"
        assert s.ntasks == 4
        assert s.nodes == 2
        assert s.time_limit == "00:30:00"

    def test_derived_paths_follow_shared_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_SHARED_DIR", "/mnt/fsx")
        s = load_settings(live=False)
        assert s.remote_engine_dir == "/mnt/fsx/engine"
        assert s.remote_dag_dir == "/mnt/fsx/dags"
        assert s.engine_binary == "/mnt/fsx/engine/bin/engine"

    def test_explicit_remote_dirs_win_over_derivation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HPCCTL_REMOTE_ENGINE_DIR", "/opt/engine")
        assert load_settings(live=False).engine_binary == "/opt/engine/bin/engine"

    def test_run_dir_default(self) -> None:
        assert load_settings(live=False).run_dir == Path("./.hpcctl-run")

    def test_engine_build_dir_default(self) -> None:
        assert load_settings(live=False).engine_build_dir == Path("./engine/build")

    def test_ssh_key_path_is_expanded(self) -> None:
        assert "~" not in load_settings(live=False).ssh_key_path

    def test_schema_path_resolves_to_the_repository_contract(self) -> None:
        assert load_settings(live=False).schema_path.name == "dag_schema.json"

    def test_empty_string_is_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_QUEUE_NAME", "   ")
        assert load_settings(live=False).queue_name == "compute"


class TestFallbacks:
    def test_aws_region_falls_back_to_aws_default_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-2")
        s = load_settings(live=False)
        assert s.region == "eu-west-2"
        assert "AWS_REGION" not in s.missing

    def test_aws_region_wins_over_the_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-2")
        assert load_settings(live=False).region == "us-east-1"

    def test_compute_subnet_falls_back_to_head_subnet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HPCCTL_HEAD_SUBNET_ID", "subnet-aaaa")
        s = load_settings(live=False)
        assert s.compute_subnet_id == "subnet-aaaa"

    def test_compute_subnet_inherits_the_head_placeholder(self) -> None:
        """A single unset variable must not produce two different placeholder texts."""
        s = load_settings(live=False)
        assert s.compute_subnet_id == placeholder("HPCCTL_HEAD_SUBNET_ID")

    def test_explicit_compute_subnet_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_HEAD_SUBNET_ID", "subnet-aaaa")
        monkeypatch.setenv("HPCCTL_COMPUTE_SUBNET_ID", "subnet-bbbb")
        assert load_settings(live=False).compute_subnet_id == "subnet-bbbb"


class TestPlaceholders:
    def test_dry_run_substitutes_a_placeholder(self) -> None:
        s = load_settings(live=False)
        assert s.key_name == "<<<UNSET:HPCCTL_KEY_NAME>>>"

    def test_placeholder_is_recorded_in_missing(self) -> None:
        assert "HPCCTL_KEY_NAME" in load_settings(live=False).missing

    def test_has_placeholders_reports_true(self) -> None:
        assert load_settings(live=False).has_placeholders

    def test_fully_configured_environment_has_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, value in {
            "AWS_REGION": "us-east-1",
            "HPCCTL_KEY_NAME": "kp",
            "HPCCTL_HEAD_SUBNET_ID": "subnet-aaaa",
            "HPCCTL_BOOTSTRAP_BUCKET": "bucket",
            "HPCCTL_HEAD_NODE_HOST": "1.2.3.4",
        }.items():
            monkeypatch.setenv(name, value)
        s = load_settings(live=False)
        assert s.missing == ()
        assert not s.has_placeholders

    def test_is_placeholder_recognises_the_format(self) -> None:
        assert is_placeholder(placeholder("ANY"))
        assert not is_placeholder("us-east-1")

    def test_placeholder_is_a_legal_yaml_scalar(self) -> None:
        """The format is deliberately parseable so the 'YAML parses' test stays meaningful."""
        import yaml

        assert yaml.safe_load(f"value: {placeholder('X')}") == {"value": "<<<UNSET:X>>>"}


class TestLiveResolution:
    def test_live_raises_when_required_variables_are_unset(self) -> None:
        with pytest.raises(MissingConfigError):
            load_settings(live=True)

    def test_live_reports_every_missing_variable_at_once(self) -> None:
        """One failure per run would mean four runs to discover four problems."""
        with pytest.raises(MissingConfigError) as excinfo:
            load_settings(live=True)
        assert set(excinfo.value.variables) == set(REQUIRED_ALWAYS)

    def test_message_lists_them(self) -> None:
        with pytest.raises(MissingConfigError, match="HPCCTL_KEY_NAME"):
            load_settings(live=True)

    def test_live_succeeds_when_all_required_are_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name, value in {
            "AWS_REGION": "us-east-1",
            "HPCCTL_KEY_NAME": "kp",
            "HPCCTL_HEAD_SUBNET_ID": "subnet-aaaa",
            "HPCCTL_BOOTSTRAP_BUCKET": "bucket",
        }.items():
            monkeypatch.setenv(name, value)
        assert load_settings(live=True).region == "us-east-1"

    def test_required_set_is_per_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A remote command must not demand the cluster-creation variables."""
        monkeypatch.setenv("HPCCTL_HEAD_NODE_HOST", "1.2.3.4")
        settings = load_settings(live=True, required=REQUIRED_FOR_REMOTE)
        assert settings.head_node_host == "1.2.3.4"

    def test_remote_command_still_fails_without_its_own_variable(self) -> None:
        with pytest.raises(MissingConfigError) as excinfo:
            load_settings(live=True, required=REQUIRED_FOR_REMOTE)
        assert excinfo.value.variables == ("HPCCTL_HEAD_NODE_HOST",)

    def test_exit_code_is_config(self) -> None:
        with pytest.raises(MissingConfigError) as excinfo:
            load_settings(live=True)
        assert int(excinfo.value.code) == 3


class TestStrict:
    def test_strict_fails_in_dry_run(self) -> None:
        with pytest.raises(MissingConfigError):
            load_settings(live=False, strict=True)

    def test_strict_passes_when_every_required_variable_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name, value in FULL_ENV.items():
            monkeypatch.setenv(name, value)
        assert load_settings(live=False, strict=True).missing == ()

    def test_strict_ignores_placeholders_outside_the_required_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HPCCTL_HEAD_NODE_HOST is recorded as missing but does not block a boot."""
        for name in REQUIRED_ALWAYS:
            monkeypatch.setenv(name, "x")
        settings = load_settings(live=False, strict=True, required=REQUIRED_ALWAYS)
        assert settings.missing == ("HPCCTL_HEAD_NODE_HOST",)

    def test_strict_hint_mentions_the_flag(self) -> None:
        with pytest.raises(MissingConfigError) as excinfo:
            load_settings(live=False, strict=True)
        assert excinfo.value.hint is not None
        assert "--strict" in excinfo.value.hint


class TestNumericParsing:
    @pytest.mark.parametrize(
        "name", ["HPCCTL_MIN_NODES", "HPCCTL_MAX_NODES", "HPCCTL_NTASKS", "HPCCTL_NODES"]
    )
    def test_non_integer_is_rejected(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(name, "lots")
        with pytest.raises(InvalidConfigError, match="must be an integer"):
            load_settings(live=False)

    def test_malformed_number_is_fatal_even_in_dry_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A present-but-wrong value is a typo, not an absent AWS account."""
        monkeypatch.setenv("HPCCTL_MAX_NODES", "4.5")
        with pytest.raises(InvalidConfigError):
            load_settings(live=False)

    def test_valid_integers_are_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_MAX_NODES", "16")
        assert load_settings(live=False).max_nodes == 16

    def test_exit_code_is_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_NODES", "x")
        with pytest.raises(InvalidConfigError) as excinfo:
            load_settings(live=False)
        assert int(excinfo.value.code) == 3


class TestGlobalSwitches:
    def test_dry_run_forced_is_false_when_unset(self) -> None:
        assert not dry_run_forced()

    @pytest.mark.parametrize("value", ["1", "true", "yes", "anything"])
    def test_dry_run_forced_by_any_non_empty_value(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HPCCTL_DRY_RUN", value)
        assert dry_run_forced()

    def test_empty_value_does_not_force(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HPCCTL_DRY_RUN", "")
        assert not dry_run_forced()

    def test_no_color_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert not color_disabled()
        monkeypatch.setenv("NO_COLOR", "1")
        assert color_disabled()


class TestImmutability:
    def test_settings_are_frozen(self, settings: Settings) -> None:
        """Commands derive variants with dataclasses.replace rather than mutating."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.cluster_name = "other"  # type: ignore[misc]
