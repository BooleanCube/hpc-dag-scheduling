"""Tests for ``--raw`` byte fidelity, and the negative control that justifies it.

:class:`TestRichIsLossy` is the reason this file exists. Rich reflows and truncates long lines,
so the pretty output of an artifact is *not* the artifact. Asserting that the rendered form
differs from the source stops a later reader from "simplifying" the raw path away on the
reasonable-sounding grounds that it duplicates the rendered one.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml
from conftest import blank_env
from typer.testing import CliRunner

from hpcctl.cli import app
from hpcctl.console import RAW_DELIMITER
from hpcctl.generators.bootstrap import bootstrap_text

DELIMITER_PATTERN = re.compile(
    r"^" + re.escape(RAW_DELIMITER).replace(re.escape("{name}"), r"(?P<name>\S+)") + r"$",
    re.M,
)


def split_artifacts(raw: str) -> dict[str, str]:
    """Split ``--raw`` output back into its constituent artifacts.

    Args:
        raw: Complete stdout from a ``--raw`` invocation.

    Returns:
        A mapping of artifact name to exact body text.
    """
    parts = DELIMITER_PATTERN.split(raw)
    names = parts[1::2]
    bodies = parts[2::2]
    return {name: body.lstrip("\n") for name, body in zip(names, bodies, strict=True)}


@pytest.fixture
def raw_boot(runner: CliRunner) -> dict[str, str]:
    """Invoke ``boot --raw`` with an empty environment and split the artifacts.

    Args:
        runner: The Typer CLI runner.

    Returns:
        A mapping of artifact name to body.
    """
    result = runner.invoke(app, ["boot", "--raw"], env=blank_env())
    assert result.exit_code == 0, result.stderr
    return split_artifacts(result.stdout)


class TestRawIsPipeable:
    def test_emits_the_expected_artifacts(self, raw_boot: dict[str, str]) -> None:
        assert set(raw_boot) == {"bootstrap", "cluster-config", "commands"}

    def test_bootstrap_is_byte_identical_to_the_packaged_script(
        self, raw_boot: dict[str, str]
    ) -> None:
        """The whole point: what you pipe is exactly what the nodes will run."""
        assert raw_boot["bootstrap"] == bootstrap_text()

    def test_bootstrap_passes_bash_syntax_check(
        self, raw_boot: dict[str, str], tmp_path: Path
    ) -> None:
        path = tmp_path / "piped.sh"
        path.write_text(raw_boot["bootstrap"], encoding="utf-8")
        result = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_config_parses_as_yaml(self, raw_boot: dict[str, str]) -> None:
        document = yaml.safe_load(raw_boot["cluster-config"])
        assert document["Scheduling"]["Scheduler"] == "slurm"

    def test_config_retains_the_full_s3_url(self, raw_boot: dict[str, str]) -> None:
        document = yaml.safe_load(raw_boot["cluster-config"])
        url = document["HeadNode"]["CustomActions"]["OnNodeConfigured"]["Script"]
        assert url.startswith("s3://")
        assert url.endswith(".sh")

    def test_stdout_carries_no_ansi_escapes(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["boot", "--raw"], env=blank_env())
        assert "\x1b[" not in result.stdout

    def test_warnings_stay_on_stderr(self, runner: CliRunner) -> None:
        """Stdout must stay clean enough to pipe even when config is incomplete."""
        result = runner.invoke(app, ["boot", "--raw"], env=blank_env())
        assert "<<<UNSET:" in result.stdout  # inside the config body, as a value
        assert split_artifacts(result.stdout)["bootstrap"] == bootstrap_text()

    def test_delimiter_is_a_comment_in_both_languages(self) -> None:
        """So a split artifact stays valid bash and valid YAML even with the marker attached."""
        assert RAW_DELIMITER.startswith("#")

    def test_submit_raw_emits_only_the_batch_script(
        self, runner: CliRunner, valid_dag: Path
    ) -> None:
        result = runner.invoke(app, ["submit", "--dag", str(valid_dag), "--raw"], env=blank_env())
        assert result.exit_code == 0
        artifacts = split_artifacts(result.stdout)
        assert set(artifacts) == {"sbatch"}

    def test_submit_raw_passes_bash_syntax_check(
        self, runner: CliRunner, valid_dag: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(app, ["submit", "--dag", str(valid_dag), "--raw"], env=blank_env())
        path = tmp_path / "piped.sbatch"
        path.write_text(split_artifacts(result.stdout)["sbatch"], encoding="utf-8")
        check = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, check=False
        )
        assert check.returncode == 0, check.stderr


class TestRichIsLossy:
    """The negative control: rendered output is not the artifact."""

    def test_rendered_bootstrap_is_not_byte_identical(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--no-color", "boot"], env=blank_env())
        assert result.exit_code == 0
        assert bootstrap_text() not in result.stdout

    def test_a_long_line_is_reflowed_by_the_renderer(self, runner: CliRunner) -> None:
        """A 224-character apt-get line does not survive rendering intact at width 80."""
        long_line = "  " + " ".join(f"package-number-{n:03d}" for n in range(12))
        assert len(long_line) > 200
        from hpcctl import console

        console.configure(no_color=True)
        with console.out().capture() as captured:
            console.render_artifact("t", f"#!/bin/bash\napt-get install -y {long_line}\n", "bash")
        rendered = captured.get()
        assert long_line not in rendered

    def test_word_wrap_preserves_content_even_though_layout_changes(
        self, runner: CliRunner
    ) -> None:
        """word_wrap means the text is reflowed, not silently truncated: no packages are lost."""
        from hpcctl import console

        console.configure(no_color=True)
        with console.out().capture() as captured:
            console.render_artifact("t", bootstrap_text(), "bash")
        rendered = captured.get()
        for package in ("nlohmann-json3-dev", "libprotobuf-dev", "protobuf-compiler"):
            assert package in rendered

    def test_raw_and_rendered_differ_for_the_same_artifact(self, runner: CliRunner) -> None:
        raw = runner.invoke(app, ["boot", "--raw"], env=blank_env()).stdout
        pretty = runner.invoke(app, ["--no-color", "boot"], env=blank_env()).stdout
        assert raw != pretty

    def test_artifacts_are_never_wrapped_in_a_panel(self) -> None:
        """Panel borders steal width and re-truncate; Rule labels do not."""
        source = (Path(__file__).resolve().parents[1] / "src" / "hpcctl" / "console.py").read_text(
            encoding="utf-8"
        )
        body = source.split("def render_artifact", 1)[1].split("\ndef ", 1)[0]
        assert "Panel(" not in body
        assert "Rule(" in body
        assert "word_wrap=True" in body
