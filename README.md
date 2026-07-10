# Two Environment Variables That Fix PyTorch Memory Creep Forever

**TL;DR:** If you load and unload large models in Python/PyTorch on Linux, your process slowly eats all available RAM and eventually gets OOM-killed. The fix is two environment variables that change how glibc allocates memory. Zero code changes. Zero performance cost. Works for any model size.

```bash
export MALLOC_MMAP_THRESHOLD_=65536
export MALLOC_TRIM_THRESHOLD_=65536
```

That's it. Read on for the data.

---

## The Problem Everyone Has

If you run a Python process that loads and unloads large ML models — diffusion models, LLMs, vision models, anything — you've probably seen this:

1. Load a model → RSS climbs to 30-40GB
2. Unload the model → RSS drops... but only to 7-10GB
3. Load a different model → RSS climbs again
4. Unload → RSS "settles" even higher than last time
5. Repeat for hours → process hits 50GB+ → Linux OOM killer terminates it

The standard advice:
- `gc.collect()` — helps Python objects, but the RSS doesn't drop
- `torch.cuda.empty_cache()` — clears GPU memory, doesn't touch system RAM
- `ctypes.CDLL("libc.so.6").malloc_trim(0)` — helps a bit, but RSS still creeps
- Restart the process periodically — works but ugly
- Use subprocess workers — works but slow
- Just add more RAM — works but expensive

None of these fix the root cause.

## The Root Cause: glibc Arena Fragmentation

When Python/PyTorch allocates memory for model weights (typically 2-30GB), glibc's default allocator uses `sbrk()` to extend the heap. This memory is allocated in **arenas** — contiguous chunks that the allocator manages internally.

The critical behavior: **glibc's heap arenas never shrink back to their original size.** When you `free()` the memory, it's marked as available within the arena, but the arena's address space is not returned to the operating system. `malloc_trim()` helps less than you'd hope: since glibc 2.8 it can release whole free pages anywhere in the heap (not just the top), but any page still containing even one small live allocation cannot be released — and Python interleaves small long-lived allocations everywhere. The fragmented arena survives every trim.

Each model load/unload cycle fragments the arena slightly differently. Over hundreds of cycles, the arena grows permanently. The memory isn't leaked — glibc knows it's free — but the OS can't reclaim it because the heap boundary never moves back.

## The Fix: Force Large Allocations Through mmap

```bash
export MALLOC_MMAP_THRESHOLD_=65536   # 64KB
export MALLOC_TRIM_THRESHOLD_=65536   # 64KB
```

`MALLOC_MMAP_THRESHOLD_` tells glibc: "for any allocation larger than 64KB, use `mmap()` instead of the heap arena."

The difference:
- **Heap (sbrk)**: Memory is part of a contiguous arena. Can only be returned to the OS if ALL memory above it is also free. Fragments permanently.
- **mmap**: Memory is mapped as independent pages. When freed with `munmap()`, the pages are **immediately and completely** returned to the OS. No fragmentation possible.

Model weights are multi-gigabyte allocations. With the threshold set to 64KB, they go through mmap. When the model is unloaded and the tensors are freed, the OS gets every single page back instantly.

## The Data

We run a render pipeline that cycles through 13 different Stable Diffusion / Flux / PixArt models on a 62GB Linux server with an AMD RX 7800 XT (16GB VRAM). Models load into CPU RAM (some use GPU offloading), render, then unload to make room for the next model.

### Without the fix (default glibc behavior)

Flux Schnell model (30GB in RAM):

| Event | RSS | Notes |
|-------|-----|-------|
| Baseline (idle) | 943 MB | Clean process start |
| After Flux load | 36,459 MB | Model weights in RAM |
| After Flux unload + gc + malloc_trim | 7,099 MB | **6.2GB stuck in arena** |
| After 2nd Flux cycle | 12,172 MB | **Creeping higher** |
| After 17 hours of cycling | 52,000 MB | **OOM killed by kernel** |

Post-unload RSS **never returns to baseline**. Each cycle permanently raises it by ~450MB. After 17 hours of continuous model switching, the process hit 52GB and the Linux OOM killer terminated it.

We instrumented the unload path with timed RSS sampling — waiting 30 seconds after unload showed **zero additional recovery**. The memory wasn't "still releasing." It was permanently trapped in the glibc arena.

### With the fix

Same workload, same models, same hardware:

| Event | RSS | Notes |
|-------|-----|-------|
| Baseline (idle) | 943 MB | Clean process start |
| After Flux load | 31,262 MB | Model weights in RAM |
| After Flux unload + gc + malloc_trim | **1,205 MB** | **FULLY RECLAIMED** |
| After SDXL load + unload | **1,348 MB** | Back to baseline |
| After 2nd Flux load + unload | **934 MB** | **Lower than starting baseline** |

Post-unload RSS returns to **~1,200MB every single time**. Zero drift. Zero fragmentation. The process can run indefinitely.

### Side-by-side

```
                    WITHOUT FIX      WITH FIX
Flux unload RSS:    7,099 MB         1,205 MB
2nd Flux unload:    12,172 MB        934 MB
After 17 hours:     52,000 MB (OOM)  ~1,200 MB (stable)
```

## How to Apply

### Systemd service

```ini
[Service]
Environment=MALLOC_MMAP_THRESHOLD_=65536
Environment=MALLOC_TRIM_THRESHOLD_=65536
ExecStart=/path/to/python model_server.py
```

### Docker

```dockerfile
ENV MALLOC_MMAP_THRESHOLD_=65536
ENV MALLOC_TRIM_THRESHOLD_=65536
```

### Command line

```bash
MALLOC_MMAP_THRESHOLD_=65536 MALLOC_TRIM_THRESHOLD_=65536 python model_server.py
```

### From inside a running process (ComfyUI, notebooks, plugins)

**⚠️ Setting `os.environ` inside Python does NOT work** — glibc reads the `MALLOC_*` variables once, at allocator initialization, which happens during interpreter startup, *before your first line of Python runs*. By the time `os.environ[...] = ...` executes, the allocator is already configured; the assignment only affects child processes. (We verified this empirically — see [`verify_fix.py`](verify_fix.py): 50×100KB mallocs go +49 to mmap with the env set at launch, +0 with `os.environ` set in-process.)

If you can't control the launch environment, use `mallopt` — it works at runtime:

```python
import ctypes
libc = ctypes.CDLL("libc.so.6")
libc.mallopt(-3, 65536)   # M_MMAP_THRESHOLD
libc.mallopt(-1, 65536)   # M_TRIM_THRESHOLD
# Call as early as possible: it only affects NEW allocations —
# arenas that already fragmented stay fragmented.
```

Treat `mallopt` as the fallback, not the preference: it can't defragment a heap that already grew, so call it before any model loads — and set the env at launch whenever you control the launcher.

### Verify it's actually active

Don't trust that it took — check from inside the process:

```bash
python verify_fix.py          # in your launch environment
```

It allocates 50 blocks sized between the fixed threshold (64KB) and glibc's default (128KB) and counts how many went to mmap via `mallinfo2`. **+45 or more → fix active. ~0 → you're still on the heap arena** (the usual cause: env vars set after process start).

### The biggest deployment gotcha: spawned workers

If your server spawns worker processes — `multiprocessing`, `subprocess`, Node `child_process.spawn`, Ray workers, a queue dispatcher launching renderers — **the workers only get the fix if it's in the parent process's environment** (children inherit env at spawn), or if it's passed explicitly into each spawn's `env`. A config file that only the parent reads, or a launch wrapper the children bypass, silently misses the workers: the server looks "deployed" while every process doing the actual model loading runs unprotected. This is the most common real-world failure shape — we hit it ourselves months after shipping the fix, when a video runner spawned by a server whose own env lacked the vars crept all night and stalled a heavy render into swap, with the vars sitting right there in three config files.

Verify against the running process, not the config:

```bash
tr '\0' '\n' < /proc/<worker-pid>/environ | grep MALLOC_
```

If that's empty, the workers didn't inherit it: set the vars on the **parent process's** environment (or pass them into each spawn's `env` explicitly) — a config file the parent merely *reads* is not inherited by anything. Then re-check, and let `python verify_fix.py` (or `verify_fix.fix_is_active()` inside the worker) settle it behaviorally.

## What It Affects

- **Any Python ML workload** that loads/unloads models: PyTorch, TensorFlow, JAX, ONNX
- **Any model serving framework**: vLLM, TGI, Triton, custom FastAPI servers
- **Any architecture**: Diffusion models (SDXL, Flux, PixArt), LLMs, vision models, embeddings
- **Any Linux system** using glibc (virtually all of them)

## Performance Impact

No measurable impact on our workloads — model serving and render pipelines, where allocations are large and infrequent. SDXL renders measured 5,296ms vs ~5,300ms without the fix; after three-plus months in production across five architectures (including our heaviest CPU-dequant lanes), render-time telemetry shows no regression attributable to the allocator change.

Two honest caveats:

1. **This pins the threshold and disables glibc's dynamic adaptation.** By default, glibc *raises* the mmap threshold (up to 32MB) when it sees large blocks freed — that's its defense against mmap churn. Workloads that allocate and free large buffers in a tight loop (not model serving — think per-iteration CPU tensor churn) pay mmap/munmap syscalls plus kernel page-zeroing on every cycle. Measure on your hottest path before shipping; for load-render-unload patterns it's a non-issue.
2. Model load times are marginally affected because mmap has slightly more syscall overhead than sbrk, but the difference is unmeasurable against multi-second load times.

If you're tempted by allocator replacement instead (jemalloc/tcmalloc via `LD_PRELOAD`): we measured jemalloc regressing heavy CPU-dequantization render lanes 5–10× on this same workload class. The env-var route gets the RAM back without touching hot-path performance.

## Fine Print

- **`MALLOC_MMAP_MAX_`** — glibc caps live mmap'd allocations at 65,536 by default; beyond that it silently falls back to the heap. Model serving never gets close, but allocation-heavy processes can raise it: `MALLOC_MMAP_MAX_=1048576`.
- **Modern interface** — on current glibc these knobs are also exposed as tunables: `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=65536:glibc.malloc.trim_threshold=65536`. Same effect; use whichever fits your deploy tooling.
- **Secure binaries** — `MALLOC_*` env vars are ignored in setuid/setgid (secure) processes. Not a concern for normal Python, but it's the kind of thing that makes a fix "mysteriously not work" in exotic setups.
- **Don't stack `MALLOC_ARENA_MAX=1` on top** — it shows up in memory-tuning threads as an extra saving, but it funnels every thread through a single arena lock and can serialize a multi-threaded server. The two thresholds here don't touch arena count; keep it that way unless you've measured.
- **glibc only** — musl (Alpine) and jemalloc/tcmalloc-linked builds have different allocators and different behavior; this fix is specifically for the default glibc `ptmalloc`.

## Why Nobody Talks About This

1. **It's a C/systems-level fix** — Python and PyTorch developers don't think in terms of glibc allocator behavior
2. **The symptom looks like a Python memory leak** — but `gc.collect()` and memory profilers show nothing leaked
3. **The symptom looks like a PyTorch bug** — but PyTorch's caching allocator is GPU-side, not CPU-side
4. **`malloc_trim()` partially works** — so people think they've addressed it when they haven't
5. **The glibc docs are dense** — `MALLOC_MMAP_THRESHOLD_` is buried in `mallopt(3)` man pages
6. **The workarounds are "good enough"** — restarting processes or adding RAM is easier than debugging glibc

## How We Found It

We built a memory profiling system that tracked RSS at every stage of the model lifecycle — load start, load complete, render start, render complete, unload start, unload complete. We stored everything in SQLite and ran a 189-render test matrix covering every model switching pattern.

The data showed:
- Waiting after unload doesn't help (RSS is flat at +0s through +30s)
- Same model repeated = minimal drift (~278MB over 20 cycles)
- Cross-architecture switching (Flux→SDXL→PixArt) = permanent arena expansion
- The drift plateaus in short tests but compounds over hours

Once we identified that the memory was in glibc's arena (not Python, not PyTorch, not the GPU), the fix was straightforward: force allocations through mmap where the OS can reclaim them.

## Production Data: 74 Days, 6,000 Cycles, 5,357 Model Switches

Since the original 189-render matrix, the fix has run 24/7 in our production render pipeline with full memory telemetry (every load, unload, and settled-RSS sample recorded to SQLite). The longitudinal picture, 2026-04-27 → 2026-07-09:

| Metric | Value |
|---|---|
| Load/unload cycles (classic load→render→unload worker) | 6,017 |
| Model switches (consecutive loads of *different* checkpoints) | 5,357 |
| Distinct checkpoints | 40 |
| Distinct architectures | 19 (SDXL, Flux, PixArt, Qwen-Image, HiDream, Wan 2.2 video, …) |
| Largest single model load | +47.9GB RSS (Qwen-Image, bf16) |
| **Median settled post-unload RSS, by month** | **Apr: 1,786MB · May: 1,777MB · Jun: 1,193MB** |
| OOM kills attributable to allocator creep | **0** |

The number that matters is the monthly median settled RSS: **flat-to-declining across 74 days** of continuous model switching. Without the fix, the same pipeline gained ~450MB per switch and OOM'd in 17 hours — at that rate, 5,357 switches is ~2.4TB of phantom RSS. With it, representative long single-process sessions:

| Session | Cycles | Distinct models | Settled RSS drift (first→last) |
|---|---|---|---|
| 2026-06-13 | 101 | 22 | **+53MB** (~0.5MB/cycle) |
| 2026-06-08 | 121 | 20 | −458MB (ended *below* start) |
| 2026-06-07 | 198 | 19 | +442MB (~2MB/cycle) |

Residual single-digit-MB/cycle drift in some long sessions traces to non-allocator sources (driver/context growth, module caches, and lanes that deliberately keep warm residents — e.g. pinned text encoders — which raise the floor *by design*). The allocator-creep signature — hundreds of MB per switch, unbounded — is gone everywhere, for months.

## Run the Benchmark Yourself

```bash
# Clone this repo
git clone https://github.com/brjen/pytorch-memory-fix.git
cd pytorch-memory-fix

# Run without fix (watch RSS grow)
python benchmark.py --no-fix

# Run with fix (watch RSS stay flat)
python benchmark.py
```

See [`benchmark.py`](benchmark.py) for the full test harness.

## Hardware Tested On

- **Server**: 62GB RAM, AMD RX 7800 XT (16GB VRAM), Linux/glibc (originally tested on glibc 2.39, currently running glibc 2.43 on an Arch-based distro)
- **Models**: originally 13 checkpoints across 5 architectures (SDXL, Flux, PixArt-Sigma, Playground V2.5, Kandinsky 3); now 40 checkpoints across 19 architectures in production
- **Workload**: Continuous model switching — load model A, render, unload, load model B, render, unload, repeat
- **Test matrix**: 189 renders across every switching pattern, plus a 104-render long-run proof (107 model switches, RSS flat)

## Status

This fix is running in production 24/7. We are actively monitoring for regressions and will update this repo with any findings.

**Verified stable as of 2026-07-09** — after the original 107-switch proof, the fix has now survived 74 days of 24/7 production: 6,000+ load/unload cycles and 5,357 model switches across 19 architectures with flat monthly median settled RSS (see Production Data above).

If you encounter any regressions or edge cases, please [open an issue](https://github.com/brjen/pytorch-memory-fix/issues).

## References

- [glibc mallopt documentation](https://man7.org/linux/man-pages/man3/mallopt.3.html)
- [MALLOC_MMAP_THRESHOLD_ in glibc](https://www.gnu.org/software/libc/manual/html_node/Malloc-Tunable-Parameters.html)
- [Understanding glibc malloc](https://sourceware.org/glibc/wiki/MallocInternals)

## License

MIT — use this however you want. If it saves you from an OOM kill, that's all the credit we need.

---

*Discovered 2026-03-24 by [Gridline Studio](https://github.com/brjen) during render pipeline optimization. The fix has been running in production on a 62GB AMD server cycling 13 different diffusion models continuously.*
