"""Exhaustive check: RTL mirror vs numpy reference for every input pair.
Usage: python3 model/exhaustive_check.py [ops e.g. 0,1,2] [a-range, default 65536]
Full run takes about 35 minutes on 2 cores."""
import sys, time; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from fp16_ref import ref_numpy
import rtl_mirror as rm
BLK = 64
B = np.tile(np.arange(65536, dtype=np.uint16), BLK)
def work(args):
    op, a0 = args
    a = np.repeat(np.arange(a0, a0+BLK, dtype=np.uint16), 65536)
    rb, rf = ref_numpy(a, B, op)
    mb, mf = rm.fpu(a, B, op)
    bad = (rb != mb) | (rf != mf)
    n = int(bad.sum())
    ex = []
    if n:
        idx = np.nonzero(bad)[0][:3]
        ex = [(hex(a[i]), hex(B[i]), hex(rb[i]), int(rf[i]), hex(mb[i]), int(mf[i])) for i in idx]
    return op, n, ex
if __name__ == "__main__":
    ops = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2").split(",")]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
    t = time.time()
    tasks = [(op, a0) for op in ops for a0 in range(0, limit, BLK)]
    tot = {op: 0 for op in ops}; exs = []
    with Pool(2) as p:
        for op, n, ex in p.imap_unordered(work, tasks):
            tot[op] += n; exs += ex
    print("ops", ops, "a-range", limit, "pairs/op", limit*65536, "mismatches", tot, "time %.0fs" % (time.time()-t))
    for e in exs[:10]: print(e)
