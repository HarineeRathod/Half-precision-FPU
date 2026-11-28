"""
Generates sim/vectors.hex for the Verilog testbench (tb/tb_fpu.v).

Each line is one 64-bit hex word followed by a comment:
    OO AAAA BBBB RRRR FF   (written without spaces)
    OO   = opcode (00 add, 01 sub, 10 mul, 11 div/reserved)
    AAAA = operand a, BBBB = operand b
    RRRR = expected result, FF = expected exception flags {NV,DZ,OF,UF,NX}
The last line is the sentinel FF.... which ends the test.

Expected values come from ref_exact() (exact rational arithmetic), which does
not share any code with the RTL.

Usage:  python3 gen_vectors.py [random_per_op] [seed]
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp16_ref import ref_exact  # noqa: E402

OPN = {0: "+", 1: "-", 2: "*", 3: "/"}

DIRECTED = [
    # ---- addition: basic, alignment, rounding ----
    (0, 0x3C00, 0x3C00, "1.0 + 1.0 = 2.0"),
    (0, 0x3C00, 0x3400, "1.0 + 0.25 (exponent difference 2)"),
    (0, 0x3E00, 0x4180, "1.5 + 2.75 = 4.25"),
    (0, 0x3C00, 0x1400, "1.0 + 2^-10 = 1 ulp above 1.0"),
    (0, 0x3C00, 0x1000, "1.0 + 2^-11: exact tie, LSB even -> stays 1.0"),
    (0, 0x3C01, 0x1000, "(1+2^-10) + 2^-11: tie, LSB odd -> rounds up"),
    (0, 0x3C00, 0x1001, "1.0 + just over half ulp -> rounds up"),
    (0, 0x6800, 0x3C00, "2048 + 1: tie -> 2048 (even)"),
    (0, 0x6800, 0x4200, "2048 + 3: tie -> 2052 (even)"),
    (0, 0x3C00, 0x0001, "1.0 + smallest subnormal: sticky only"),
    # ---- addition: overflow ----
    (0, 0x7BFF, 0x7BFF, "65504 + 65504 -> overflow to +inf"),
    (0, 0x7BFF, 0x4C00, "65504 + 16: tie at 65520 rounds up -> overflow"),
    (0, 0x7BFF, 0x4B80, "65504 + 15 -> 65504 (inexact)"),
    (0, 0xFBFF, 0xFBFF, "-65504 + -65504 -> -inf"),
    # ---- addition: subnormals ----
    (0, 0x0001, 0x0001, "min subnormal + min subnormal"),
    (0, 0x03FF, 0x0001, "max subnormal + min subnormal = min normal"),
    (0, 0x0200, 0x0200, "subnormal + subnormal -> normal"),
    (0, 0x0400, 0x8001, "min normal - min subnormal = max subnormal"),
    # ---- addition: cancellation and signed zeros ----
    (0, 0x3C00, 0xBBFF, "1.0 - 0.99951: massive cancellation, exact"),
    (0, 0x4248, 0xC248, "x + (-x) = +0"),
    (0, 0x0000, 0x8000, "+0 + -0 = +0"),
    (0, 0x8000, 0x8000, "-0 + -0 = -0"),
    (0, 0x8000, 0x3C00, "-0 + 1.0 = 1.0"),
    # ---- addition: specials ----
    (0, 0x7C00, 0x3C00, "+inf + 1 = +inf"),
    (0, 0x7C00, 0x7C00, "+inf + +inf = +inf"),
    (0, 0x7C00, 0xFC00, "+inf + -inf = NaN, invalid"),
    (0, 0x7E00, 0x3C00, "qNaN + 1 = qNaN, no flag"),
    (0, 0x7D00, 0x3C00, "sNaN + 1 = qNaN, invalid"),
    (0, 0x0000, 0x7E00, "0 + qNaN = qNaN"),
    (0, 0xFE55, 0x7C00, "negative NaN with payload + inf = canonical qNaN"),
    # ---- subtraction ----
    (1, 0x4200, 0x4500, "3 - 5 = -2"),
    (1, 0x3C00, 0x3C00, "1 - 1 = +0"),
    (1, 0xBC00, 0xBC00, "-1 - (-1) = +0"),
    (1, 0x8000, 0x0000, "-0 - +0 = -0"),
    (1, 0x0000, 0x0000, "+0 - +0 = +0"),
    (1, 0x7C00, 0x7C00, "inf - inf = NaN, invalid"),
    (1, 0x7C00, 0xFC00, "inf - (-inf) = +inf"),
    (1, 0x3C01, 0x3C00, "(1+2^-10) - 1 = 2^-10"),
    (1, 0xFBFF, 0x7BFF, "-65504 - 65504 -> -inf"),
    (1, 0x0001, 0x0002, "subnormal - subnormal = -min subnormal"),
    # ---- multiplication ----
    (2, 0x4000, 0x4200, "2 * 3 = 6"),
    (2, 0xBE00, 0x4000, "-1.5 * 2 = -3"),
    (2, 0x3C00, 0x57B0, "1.0 * x = x"),
    (2, 0x3C01, 0x3C01, "(1+2^-10)^2 -> rounds to 1+2^-9 (inexact)"),
    (2, 0x5C00, 0x5C00, "256 * 256 = 65536 -> overflow to +inf"),
    (2, 0x7BFF, 0x3C01, "65504 * (1+2^-10) -> overflow"),
    (2, 0x7BFF, 0x3BFF, "65504 * 0.99951 -> rounds, stays finite"),
    (2, 0x0400, 0x3800, "2^-14 * 0.5 = 2^-15: exact subnormal, no underflow flag"),
    (2, 0x0400, 0x0400, "2^-14 * 2^-14 -> 0, underflow + inexact"),
    (2, 0x0001, 0x3800, "min subnormal * 0.5: tie -> 0 (even)"),
    (2, 0x0001, 0x3A00, "min subnormal * 0.75 -> rounds up to min subnormal"),
    (2, 0x0001, 0x6400, "2^-24 * 1024 = 2^-14 (subnormal in, normal out)"),
    (2, 0x0001, 0x7BFF, "min subnormal * 65504: big normalization shift"),
    (2, 0x03FF, 0x4000, "max subnormal * 2 -> normal"),
    (2, 0x0155, 0x0155, "subnormal * subnormal -> 0, underflow"),
    (2, 0x0001, 0x7C00, "min subnormal * inf = inf"),
    (2, 0x7C00, 0x0000, "inf * 0 = NaN, invalid"),
    (2, 0x8000, 0xFC00, "-0 * -inf = NaN, invalid"),
    (2, 0xFC00, 0x4000, "-inf * 2 = -inf"),
    (2, 0x0000, 0xC200, "0 * -3 = -0"),
    (2, 0x7E00, 0x0000, "qNaN * 0 = qNaN, no flag"),
    (2, 0x7C01, 0x3C00, "sNaN * 1 = qNaN, invalid"),
    (2, 0x8400, 0x3400, "-2^-14 * 0.25 -> negative subnormal"),
    # ---- reserved opcode ----
    (3, 0x3C00, 0x3C00, "opcode 11 (division) reserved -> qNaN, invalid"),
]


def interesting(rng):
    """Random operand biased toward corner regions."""
    s = rng.getrandbits(1) << 15
    r = rng.random()
    if r < 0.25:   e = 0                              # zero / subnormal
    elif r < 0.35: e = 31                             # inf / NaN
    elif r < 0.50: e = rng.choice([1, 2, 29, 30])     # edges of the normal range
    else:          e = rng.randint(1, 30)
    f = rng.choice([0, 1, 0x3FF, 0x200, 0x1FF, rng.getrandbits(10), rng.getrandbits(10)])
    return s | (e << 10) | f


def near(rng, x):
    """Operand with an exponent close to x's (exercises cancellation / rounding)."""
    e = min(30, max(0, ((x >> 10) & 31) + rng.randint(-2, 2)))
    return (rng.getrandbits(1) << 15) | (e << 10) | rng.getrandbits(10)


def main():
    n_rand = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2027
    rng = random.Random(seed)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "sim", "vectors.hex")

    vecs = [(op, a, b, c) for op, a, b, c in DIRECTED]
    for op in (0, 1, 2):
        for _ in range(n_rand):
            vecs.append((op, rng.getrandbits(16), rng.getrandbits(16), "random"))
        for _ in range(n_rand):
            a = interesting(rng)
            b = near(rng, a) if rng.random() < 0.5 else interesting(rng)
            vecs.append((op, a, b, "corner-biased random"))
    for _ in range(50):
        vecs.append((3, rng.getrandbits(16), rng.getrandbits(16), "reserved opcode"))

    with open(out, "w") as fh:
        for op, a, b, c in vecs:
            r, fl = ref_exact(a, b, op)
            fh.write("%02X%04X%04X%04X%02X  // %04X %s %04X = %04X flags=%s  %s\n" %
                     (op, a, b, r, fl, a, OPN[op], b, r, format(fl, "05b"), c))
        fh.write("FF00000000000000  // end of vectors\n")
    print("wrote %d vectors to %s" % (len(vecs), out))


if __name__ == "__main__":
    main()
