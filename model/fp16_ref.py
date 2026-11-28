"""
Golden reference models for the binary16 FPU.

Two independent references:
  ref_exact()  - scalar, exact rational arithmetic (fractions.Fraction) plus a
                 hand-written IEEE 754 round-to-nearest-even. Used to create the
                 Verilog test vectors.
  ref_numpy()  - vectorized: exact float64 arithmetic, then numpy's float64->float16
                 conversion (correctly rounded). Used for exhaustive checking.

Both return (result_bits, flags) where flags = {NV, DZ, OF, UF, NX} packed as
bit4..bit0, NaN results are canonical 0x7E00, and tininess is detected before
rounding (result magnitude < 2^-14 with unbounded exponent).
"""
from fractions import Fraction
import numpy as np

QNAN = 0x7E00
OP_ADD, OP_SUB, OP_MUL, OP_DIV = 0, 1, 2, 3
NV, DZ, OF, UF, NX = 16, 8, 4, 2, 1
MIN_NORMAL = Fraction(1, 2 ** 14)


# ----------------------------------------------------------------------------- scalar
def is_nan(x):  return (x >> 10) & 31 == 31 and x & 0x3FF != 0
def is_snan(x): return is_nan(x) and not (x >> 9) & 1
def is_inf(x):  return (x >> 10) & 31 == 31 and x & 0x3FF == 0


def to_fraction(x):
    """Finite binary16 bit pattern -> exact Fraction (sign of zero is lost)."""
    s, e, f = x >> 15, (x >> 10) & 31, x & 0x3FF
    v = Fraction(f, 1024) * Fraction(2) ** -14 if e == 0 else (1 + Fraction(f, 1024)) * Fraction(2) ** (e - 15)
    return -v if s else v


def round_to_fp16(v, sign_if_zero):
    """Exact Fraction -> (bits, OF, UF, NX) with round-to-nearest-even."""
    if v == 0:
        return (sign_if_zero << 15), 0, 0, 0
    s = 1 if v < 0 else 0
    m = abs(v)
    # find e such that 2^e <= m < 2^(e+1)
    e = m.numerator.bit_length() - m.denominator.bit_length()
    if Fraction(2) ** e > m:
        e -= 1
    tiny = m < MIN_NORMAL
    q = max(e, -14) - 10                      # weight of the result LSB
    scaled = m / Fraction(2) ** q             # value in units of the LSB
    n = scaled.numerator // scaled.denominator
    rem = scaled - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n & 1):
        n += 1
    inexact = rem != 0
    # n is the integer significand at weight 2^q
    if n == 2048:                             # rounded up to the next binade
        n, q = 1024, q + 1
    if n >= 1024:
        biased = q + 10 + 15
        if biased >= 31:
            return (s << 15) | 0x7C00, 1, 0, 1   # overflow -> infinity
        bits = (s << 15) | (biased << 10) | (n - 1024)
    else:
        bits = (s << 15) | n                  # subnormal (q == -24)
    return bits, 0, int(tiny and inexact), int(inexact)


def ref_exact(a, b, op):
    if op == OP_DIV:
        return QNAN, NV                       # division not implemented: reserved opcode
    if op == OP_SUB:
        b ^= 0x8000
    nv = NV if (is_snan(a) or is_snan(b)) else 0
    if is_nan(a) or is_nan(b):
        return QNAN, nv
    sa, sb = a >> 15, b >> 15
    if op in (OP_ADD, OP_SUB):
        if is_inf(a) and is_inf(b):
            return (a, 0) if sa == sb else (QNAN, NV)
        if is_inf(a): return a, 0
        if is_inf(b): return b, 0
        v = to_fraction(a) + to_fraction(b)
        zero_sign = sa & sb                   # RNE: -0 only for (-0) + (-0)
    else:
        rs = sa ^ sb
        a_zero, b_zero = (a & 0x7FFF) == 0, (b & 0x7FFF) == 0
        if (is_inf(a) and b_zero) or (is_inf(b) and a_zero):
            return QNAN, NV
        if is_inf(a) or is_inf(b):
            return (rs << 15) | 0x7C00, 0
        v = to_fraction(a) * to_fraction(b)
        zero_sign = rs
    bits, of, uf, nx = round_to_fp16(v, zero_sign)
    if v != 0 and (bits & 0x7FFF) == 0:       # underflowed to zero keeps the exact result's sign
        bits = (1 if v < 0 else 0) << 15
    return bits, (OF if of else 0) | (UF if uf else 0) | (NX if nx else 0)


# ----------------------------------------------------------------------------- vectorized
def ref_numpy(a, b, op):
    a = np.asarray(a, dtype=np.uint16)
    b = np.asarray(b, dtype=np.uint16)
    if op == OP_SUB:
        b = b ^ np.uint16(0x8000)
    fa = a.view(np.float16).astype(np.float64)
    fb = b.view(np.float16).astype(np.float64)
    with np.errstate(all="ignore"):
        exact = fa + fb if op in (OP_ADD, OP_SUB) else fa * fb   # exact in float64
        r16 = exact.astype(np.float16)                             # single correct rounding
    bits = r16.view(np.uint16).astype(np.int64)
    nan_out = np.isnan(r16)
    bits = np.where(nan_out, QNAN, bits)

    def snan(x):
        x = x.astype(np.int64)
        return (((x >> 10) & 31) == 31) & ((x & 0x3FF) != 0) & (((x >> 9) & 1) == 0)

    in_nan = np.isnan(fa) | np.isnan(fb)
    finite_in = np.isfinite(fa) & np.isfinite(fb)
    nv = snan(a) | snan(b) | (nan_out & ~in_nan)
    with np.errstate(all="ignore"):
        nx = finite_in & (r16.astype(np.float64) != exact)
        of = finite_in & np.isinf(r16)
        uf = finite_in & (exact != 0) & (np.abs(exact) < 2.0 ** -14) & nx
    flags = nv * NV + of * OF + uf * UF + nx * NX
    return bits, flags.astype(np.int64)
