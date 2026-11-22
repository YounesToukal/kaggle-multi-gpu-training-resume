# Cell 2 - carry the previous version's run forward, then train on every GPU (put it right after the %%writefile cell)
import os, glob, shutil, subprocess, sys, time, torch
SESSION_T0 = time.time()                          # Kaggle's 12 hours run from the start of the session
BUDGET_H = 11.25                                  # stop well before 12 h, so the save and Kaggle's copy fit
INPUT = os.environ.get("INPUT", "/kaggle/input")  # (the env lookups only make the cell testable off Kaggle)
WORK = os.environ.get("WORK", "/kaggle/working")
RUN = os.path.join(WORK, "run")

# 1. the previous version's output, attached as an input, holds run/state.pt: copy the most advanced one
prev = glob.glob(os.path.join(INPUT, "**", "run", "state.pt"), recursive=True)
if prev and not os.path.exists(os.path.join(RUN, "state.pt")):
    best = max(prev, key=lambda p: torch.load(p, map_location="cpu", weights_only=False)["step"])
    shutil.copytree(os.path.dirname(best), RUN, dirs_exist_ok=True)
    print("continuing from", best)

# 2. one process per GPU; progress goes to a log file, printed here every minute
n_gpu = max(torch.cuda.device_count(), 1)
cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpu}", "train.py",
       "--out", RUN, "--deadline", str(SESSION_T0 + BUDGET_H * 3600)] + os.environ.get("EXTRA", "").split()
with open(os.path.join(WORK, "train.log"), "a") as log:
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, NCCL_P2P_DISABLE="1"))
    while p.poll() is None:
        time.sleep(int(os.environ.get("POLL", "60")))
        lines = open(os.path.join(WORK, "train.log")).read().splitlines()
        print(lines[-1] if lines else "starting...", flush=True)
print(open(os.path.join(WORK, "train.log")).read().splitlines()[-1])   # ALL DONE, or NOT FINISHED
