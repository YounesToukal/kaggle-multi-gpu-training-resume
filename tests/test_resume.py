import json, os, subprocess, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def train(out, *extra, nproc=2):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
           "train.py", "--out", str(out), *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout


def launch(work, inp, extra=""):
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ, WORK=str(work), INPUT=str(inp), EXTRA=extra, POLL="1")
    subprocess.run([sys.executable, "launch.py"], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return (Path(work) / "train.log").read_text().splitlines()[-1]


def state(run):
    return torch.load(Path(run) / "state.pt", map_location="cpu", weights_only=False)


def assert_same(a, b):
    assert a["step"] == b["step"]
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    for k, v in a["opt"]["state"].items():
        for name in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(v[name], b["opt"]["state"][k][name]), (k, name)


def test_paused_twice_matches_uninterrupted(tmp_path):
    assert "ALL DONE" in train(tmp_path / "full")
    assert "NOT FINISHED" in train(tmp_path / "paused", "--max-steps-now", "20")
    assert "NOT FINISHED" in train(tmp_path / "paused", "--max-steps-now", "30")    # resumes mid-epoch
    assert "ALL DONE" in train(tmp_path / "paused")
    assert_same(state(tmp_path / "full"), state(tmp_path / "paused"))


def test_launcher_carries_run_across_versions(tmp_path):
    empty = tmp_path / "no-input"
    empty.mkdir()
    assert launch(tmp_path / "full", empty) == "ALL DONE"
    # version 1 pauses; its output is then attached to version 2 as an input
    assert launch(tmp_path / "input" / "version-1", empty, "--max-steps-now 20").startswith("NOT FINISHED")
    assert launch(tmp_path / "version-2", tmp_path / "input") == "ALL DONE"
    assert_same(state(tmp_path / "full" / "run"), state(tmp_path / "version-2" / "run"))


def test_notebook_matches_scripts():
    cells = [c for c in json.loads((ROOT / "kaggle_template.ipynb").read_text())["cells"] if c["cell_type"] == "code"]
    assert "".join(cells[0]["source"]) == "%%writefile train.py\n" + (ROOT / "train.py").read_text()
    assert "".join(cells[1]["source"]) == (ROOT / "launch.py").read_text()
