# train.py - one training run on all GPUs of the machine, saved every SAVE_EVERY steps, resumable anywhere
import os, time, math, argparse
import numpy as np, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/kaggle/working/run")
ap.add_argument("--deadline", type=float, default=0)       # unix time by which this session must stop
ap.add_argument("--max-steps-now", type=int, default=0)     # for testing resume: pause after this many steps
args = ap.parse_args()

EPOCHS, BATCH, LR, SEED = 3, 32, 1e-3, 42                   # BATCH = per GPU
SAVE_EVERY = 10                                             # optimiser steps between saves (a "block")
FIRST_BLOCK_MIN = 5.0                                       # your guess for one block, until one is measured
WORKERS = int(os.environ.get("WORKERS", "0"))

# ---------- who am I ----------
RANK, WORLD, LOCAL = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
cuda = torch.cuda.is_available()
dev = torch.device("cuda", LOCAL) if cuda else torch.device("cpu")
if cuda:
    torch.cuda.set_device(LOCAL)
dist.init_process_group("nccl" if cuda else "gloo")
os.makedirs(args.out, exist_ok=True)
STATE = os.path.join(args.out, "state.pt")                  # everything, in one file

def say(*a):
    if RANK == 0:
        print(*a, flush=True)

def agree(flag):
    """process 0 decides, everyone follows: no process waits for a partner that already left"""
    t = torch.tensor([int(flag)], device=dev)
    dist.broadcast(t, src=0)
    return bool(t.item())

# ---------- YOUR DATA (a toy problem here): augmentation seeded per sample, so a resumed run sees the same samples
class Toy(Dataset):
    def __init__(self, n=2048):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 32, generator=g)
        self.y = (self.x[:, :4].sum(1, keepdim=True) > 0).float()
        self.epoch = 0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        rng = np.random.default_rng((SEED, self.epoch, i))   # the same noise for this sample in this epoch, always
        return self.x[i] + torch.from_numpy(rng.normal(0, 0.1, 32)).float(), self.y[i]

data = Toy()

# ---------- YOUR MODEL, optimiser and schedule ----------
torch.manual_seed(SEED)                                      # the same initial weights on every GPU
model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)).to(dev)
sampler = DistributedSampler(data, num_replicas=WORLD, rank=RANK, shuffle=True, seed=SEED, drop_last=True)
STEPS_PER_EPOCH = len(sampler) // BATCH                      # the same on every GPU (drop_last)
TOTAL = EPOCHS * STEPS_PER_EPOCH
opt = torch.optim.AdamW(model.parameters(), lr=LR)
warm = max(1, TOTAL // 20)                                   # linear warm-up, then cosine decay, per step
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, TOTAL - warm))))
scaler = torch.amp.GradScaler("cuda", enabled=cuda)
loss_fn = nn.BCEWithLogitsLoss()

# ---------- resume ----------
step, block_min = 0, []                                      # step = optimiser steps done in the whole run
if os.path.exists(STATE):
    S = torch.load(STATE, map_location="cpu", weights_only=False)
    model.load_state_dict(S["model"]); opt.load_state_dict(S["opt"])
    sched.load_state_dict(S["sched"]); scaler.load_state_dict(S["scaler"])
    step, block_min = S["step"], S["block_min"]
    R = S["rng"][RANK]                                        # this GPU's random state, saved with the rest
    torch.set_rng_state(R["cpu"])
    if cuda:
        torch.cuda.set_rng_state(R["cuda"])
    say(f"resumed at step {step} of {TOTAL}")
if step == 0:
    torch.manual_seed(SEED + 1000 * RANK)                    # dropout: different on each GPU, reproducible
ddp = DDP(model, device_ids=[LOCAL] if cuda else None)

def save():
    mine = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state() if cuda else None}
    rngs = [None] * WORLD
    dist.all_gather_object(rngs, mine)                       # every GPU's random state, collected on every process
    if RANK == 0:                                            # one process writes ONE file with everything
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "block_min": block_min, "rng": rngs}, STATE + ".tmp")
        os.replace(STATE + ".tmp", STATE)                    # atomic: the old state or the new one, never a mix
    dist.barrier()

# ---------- train ----------
t_block, stopped, done_now = time.time(), False, 0
while step < TOTAL and not stopped:
    epoch, first = divmod(step, STEPS_PER_EPOCH)
    sampler.set_epoch(epoch); data.epoch = epoch             # this epoch's shuffle, the same on every run
    order = list(sampler)[first * BATCH:STEPS_PER_EPOCH * BATCH]   # skip the batches already trained
    loader = DataLoader(data, batch_size=BATCH, sampler=order, num_workers=WORKERS,
                        generator=torch.Generator().manual_seed(SEED + epoch))   # keeps the global RNG untouched
    ddp.train()
    for x, y in loader:
        if step % SAVE_EVERY == 0:                           # a new block starts: will it finish in time?
            est = float(np.median(block_min[-3:])) if block_min else FIRST_BLOCK_MIN
            fits = args.deadline == 0 or time.time() + 1.1 * est * 60 < args.deadline
            if args.max_steps_now and done_now >= args.max_steps_now:
                fits = False
            if not agree(fits):
                stopped = True
                break
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda" if cuda else "cpu", enabled=cuda):
            loss = loss_fn(ddp(x), y)
        bad = torch.tensor([int(not torch.isfinite(loss))], device=dev)
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)           # a NaN on any GPU skips the step on all
        if not bad.item():
            scaler.scale(loss).backward()                    # DDP averages the gradients of all GPUs here
            scaler.step(opt); scaler.update()
        sched.step()
        step += 1; done_now += 1
        if step % SAVE_EVERY == 0 or step == TOTAL:
            block_min.append((time.time() - t_block) / 60)
            save()
            t_block = time.time()
            say(f"step {step}/{TOTAL} (epoch {step / STEPS_PER_EPOCH:.2f}) loss {loss.item():.4f} saved")
say("ALL DONE" if step == TOTAL else f"NOT FINISHED: paused at step {step} of {TOTAL}; run the next version")
dist.destroy_process_group()
