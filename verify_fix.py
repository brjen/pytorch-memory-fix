#!/usr/bin/env python3
"""Verify whether the MALLOC_MMAP_THRESHOLD_ fix is ACTIVE in this process.

glibc reads MALLOC_* env vars once, at allocator initialization — before any
Python code runs. Setting os.environ inside Python therefore does nothing for
the current process (a very common mistake, e.g. in ComfyUI custom nodes).
This script checks the allocator's actual behavior, not the environment.

Method: allocate 50 blocks of 100KB — larger than the fixed threshold (64KB),
smaller than glibc's default (128KB) — and count how many were served by mmap
using mallinfo2(). With the fix active they go to mmap; without it they land
in the heap arena.

Usage:
    python verify_fix.py                 # standalone check of your launch env
    from verify_fix import fix_is_active # or call inside any running process
"""
import ctypes
import sys


class _MallInfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd",
        "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


def fix_is_active(verbose=False):
    """Return True if allocations >64KB are being served by mmap."""
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        raise SystemExit("Not a glibc system — this fix does not apply.")
    if not hasattr(libc, "mallinfo2"):
        raise SystemExit("glibc too old for mallinfo2 (need 2.33+); "
                         "check /proc/self/smaps manually instead.")
    libc.mallinfo2.restype = _MallInfo2
    libc.malloc.restype = ctypes.c_void_p

    before = libc.mallinfo2().hblks
    ptrs = [libc.malloc(100 * 1024) for _ in range(50)]
    after = libc.mallinfo2().hblks
    for p in ptrs:
        libc.free(ctypes.c_void_p(p))

    delta = after - before
    active = delta >= 45
    if verbose:
        print(f"mmap'd blocks for 50 x 100KB mallocs: {delta:+d}")
        if active:
            print("ACTIVE — allocations >64KB are going through mmap; "
                  "freed model memory returns to the OS immediately.")
        else:
            print("NOT ACTIVE — allocations are landing in the heap arena.")
            print("Most common cause: MALLOC_MMAP_THRESHOLD_ was set after "
                  "process start (os.environ inside Python does not work).")
            print("Fix: set the env vars in the shell/service that LAUNCHES "
                  "the process, or call at runtime:")
            print('  ctypes.CDLL("libc.so.6").mallopt(-3, 65536)  # M_MMAP_THRESHOLD')
    return active


if __name__ == "__main__":
    sys.exit(0 if fix_is_active(verbose=True) else 1)
