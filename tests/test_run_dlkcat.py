"""Tests for the DLKcat runner. All Docker invocations are mocked."""
import subprocess
from pathlib import Path

import pytest

from geckopy.gather_kcats.run_dlkcat import _run_dlkcat


# --------------------------------------------------------------------------- #
# Helpers: a stub for subprocess.run that synthesizes a DLKcat output file
# --------------------------------------------------------------------------- #

class _RunSpy:
    """Captures subprocess.run calls and produces controllable behaviour.

    `version_returncode`: exit code for the `docker --version` call.
    `dlkcat_returncode`: exit code for the `docker run ...` call.
    `dlkcat_output_text`: content to write to the output file (or None to
        skip writing it, simulating "completed but no output").
    `dlkcat_stderr`: stderr text returned by the docker run.
    `raise_on_version`: if set, this exception is raised instead of
        returning a result for the `docker --version` call.
    """

    def __init__(
        self,
        *,
        version_returncode: int = 0,
        dlkcat_returncode: int = 0,
        dlkcat_output_text: str | None = "rxn1\tg1\talpha\tCC\tMASEQ\t5.0\n",
        dlkcat_stderr: str = "",
        raise_on_version: Exception | None = None,
    ):
        self.version_returncode = version_returncode
        self.dlkcat_returncode = dlkcat_returncode
        self.dlkcat_output_text = dlkcat_output_text
        self.dlkcat_stderr = dlkcat_stderr
        self.raise_on_version = raise_on_version
        self.calls: list[list[str]] = []

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(list(cmd))
        if cmd[:2] == ["docker", "--version"]:
            if self.raise_on_version is not None:
                raise self.raise_on_version
            return subprocess.CompletedProcess(
                cmd, self.version_returncode,
                stdout="Docker version 24.0.0, build deadbeef\n",
                stderr="",
            )
        # Otherwise it's the docker run; simulate by writing the output
        # file inside the mounted temp directory.
        if self.dlkcat_output_text is not None:
            # The mounted dir is the value after `-v` and before `:/data`.
            v_idx = cmd.index("-v")
            mount_spec = cmd[v_idx + 1]
            host_dir = Path(mount_spec.split(":/data")[0])
            (host_dir / "DLKcatOutput.tsv").write_text(
                self.dlkcat_output_text, encoding="utf-8",
            )
        return subprocess.CompletedProcess(
            cmd, self.dlkcat_returncode,
            stdout="", stderr=self.dlkcat_stderr,
        )


def _write_input(tmp_path: Path, name: str = "in.tsv") -> Path:
    p = tmp_path / name
    p.write_text("rxn1\tg1\talpha\tCC\tMASEQ\tNA\n", encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Pre-flight: input file existence
# --------------------------------------------------------------------------- #

def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="DLKcat input file not found"):
        _run_dlkcat(tmp_path / "missing.tsv")


# --------------------------------------------------------------------------- #
# Pre-flight: Docker availability
# --------------------------------------------------------------------------- #

def test_docker_not_installed_raises(tmp_path, monkeypatch):
    spy = _RunSpy(raise_on_version=FileNotFoundError("docker not on PATH"))
    monkeypatch.setattr(subprocess, "run", spy)
    in_path = _write_input(tmp_path)
    with pytest.raises(RuntimeError, match="Cannot find Docker"):
        _run_dlkcat(in_path)


def test_docker_version_nonzero_raises(tmp_path, monkeypatch):
    spy = _RunSpy(version_returncode=1)
    monkeypatch.setattr(subprocess, "run", spy)
    in_path = _write_input(tmp_path)
    with pytest.raises(RuntimeError, match="Docker is installed but"):
        _run_dlkcat(in_path)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

def test_happy_path_overwrites_input_by_default(tmp_path, monkeypatch):
    spy = _RunSpy()
    monkeypatch.setattr(subprocess, "run", spy)
    in_path = _write_input(tmp_path)
    original_input = in_path.read_text(encoding="utf-8")

    out = _run_dlkcat(in_path)

    assert out == in_path
    new_content = in_path.read_text(encoding="utf-8")
    assert new_content != original_input
    assert "5.0" in new_content


def test_happy_path_writes_to_explicit_output_path(tmp_path, monkeypatch):
    spy = _RunSpy()
    monkeypatch.setattr(subprocess, "run", spy)
    in_path = _write_input(tmp_path)
    out_path = tmp_path / "predictions.tsv"

    result = _run_dlkcat(in_path, output_path=out_path)

    assert result == out_path
    assert out_path.is_file()
    # Input is left alone.
    assert in_path.read_text(encoding="utf-8").endswith("NA\n")


def test_explicit_output_path_creates_parent_dir(tmp_path, monkeypatch):
    spy = _RunSpy()
    monkeypatch.setattr(subprocess, "run", spy)
    in_path = _write_input(tmp_path)
    out_path = tmp_path / "newdir" / "predictions.tsv"

    result = _run_dlkcat(in_path, output_path=out_path)

    assert result == out_path
    assert out_path.is_file()


def test_invokes_default_image(tmp_path, monkeypatch):
    spy = _RunSpy()
    monkeypatch.setattr(subprocess, "run", spy)
    in_path = _write_input(tmp_path)

    _run_dlkcat(in_path)

    docker_run_calls = [
        c for c in spy.calls
        if c[:2] == ["docker", "run"]
    ]
    assert len(docker_run_calls) == 1
    assert "ghcr.io/sysbiochalmers/dlkcat-gecko:0.1" in docker_run_calls[0]


def test_invokes_custom_image_when_specified(tmp_path, monkeypatch):
    spy = _RunSpy()
    monkeypatch.setattr(subprocess, "run", spy)
    in_path = _write_input(tmp_path)

    _run_dlkcat(in_path, docker_image="my/custom:tag")

    docker_run_calls = [
        c for c in spy.calls
        if c[:2] == ["docker", "run"]
    ]
    assert "my/custom:tag" in docker_run_calls[0]


# --------------------------------------------------------------------------- #
# Failure paths from the Docker run
# --------------------------------------------------------------------------- #

def test_docker_run_nonzero_exit_raises(tmp_path, monkeypatch):
    spy = _RunSpy(
        dlkcat_returncode=1,
        dlkcat_stderr="DLKcat exploded",
        dlkcat_output_text=None,
    )
    monkeypatch.setattr(subprocess, "run", spy)
    in_path = _write_input(tmp_path)
    with pytest.raises(RuntimeError, match="DLKcat Docker run failed"):
        _run_dlkcat(in_path)


def test_docker_run_succeeds_but_no_output_raises(tmp_path, monkeypatch):
    spy = _RunSpy(dlkcat_output_text=None)  # exit 0 but no file written
    monkeypatch.setattr(subprocess, "run", spy)
    in_path = _write_input(tmp_path)
    with pytest.raises(RuntimeError, match="produced no output file"):
        _run_dlkcat(in_path)


def test_docker_run_failure_does_not_overwrite_input(tmp_path, monkeypatch):
    """A failed run must leave the original input file untouched."""
    spy = _RunSpy(
        dlkcat_returncode=1,
        dlkcat_stderr="boom",
        dlkcat_output_text=None,
    )
    monkeypatch.setattr(subprocess, "run", spy)
    in_path = _write_input(tmp_path)
    original = in_path.read_text(encoding="utf-8")

    with pytest.raises(RuntimeError):
        _run_dlkcat(in_path)

    # Input is unchanged.
    assert in_path.read_text(encoding="utf-8") == original


# --------------------------------------------------------------------------- #
# Mount semantics
# --------------------------------------------------------------------------- #

def test_mounts_a_temp_dir_not_the_project_folder(tmp_path, monkeypatch):
    """The Docker mount must use a temp dir, not `tmp_path` itself."""
    spy = _RunSpy()
    monkeypatch.setattr(subprocess, "run", spy)
    in_path = _write_input(tmp_path)

    _run_dlkcat(in_path)

    docker_run_call = [
        c for c in spy.calls if c[:2] == ["docker", "run"]
    ][0]
    v_idx = docker_run_call.index("-v")
    mount_spec = docker_run_call[v_idx + 1]
    host_dir, container_dir = mount_spec.split(":/data")
    assert container_dir == ""  # split on full token leaves empty after
    assert "geckopy_dlkcat_" in host_dir or "tmp" in host_dir
    assert str(tmp_path) not in host_dir
