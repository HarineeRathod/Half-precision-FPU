"""
Bit-exact, vectorized Python mirror of the Verilog datapath in rtl/.
Every signal keeps the same name and bit width as in the RTL (widths enforced
with masks), so a mismatch here points at the same line in the Verilog.
"""
import numpy as np

QNAN = 0x7E00


def m(x, w):                       # truncate to w bits, like a Verilog wire [w-1:0]
    return x & ((1 << w) - 1)


def classify(x):
    e = (x >> 10) & 31
    f = x & 0x3FF
    e_max, e_zero, f_zero = e == 31, e == 0, f == 0
    return dict(
        sign=(x >> 15) & 1,
        exp=np.where(e_zero, 1, e),
        sig=np.where(e_zero, 0, 1024) | f,
        snan=e_max & ~f_zero & (((f >> 9) & 1) == 0),
        qnan=e_max & (((f >> 9) & 1) == 1),
        inf=e_max & f_zero,
        zero=e_zero & f_zero,
    )


def round_pack(sign, exp_in, mant_in, guard, sticky):
    round_up = guard & (sticky | (mant_in & 1))
    mant_rnd = m(mant_in + round_up, 12)
    carry = (mant_rnd >> 11) & 1
    mant_fin = np.where(carry == 1, m(mant_rnd >> 1, 11), m(mant_rnd, 11))
    exp_fin = m(exp_in + carry, 8)
    lost = guard | sticky
    tiny = ((mant_in >> 10) & 1) == 0
    overflow = exp_fin >= 31
    inexact = (lost == 1) | overflow
    underflow = tiny & (lost == 1)
    result = np.where(overflow, (sign << 15) | 0x7C00,
             np.where(((mant_fin >> 10) & 1) == 1,
                      (sign << 15) | (m(exp_fin, 5) << 10) | m(mant_fin, 10),
                      (sign << 15) | m(mant_fin, 10)))
    return result, overflow, underflow, inexact


def addsub(a, b, sub):
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    b_eff = b ^ (sub << 15)
    A, B = classify(a), classify(b_eff)

    swap = (b_eff & 0x7FFF) > (a & 0x7FFF)
    sel = lambda k: (np.where(swap, B[k], A[k]), np.where(swap, A[k], B[k]))
    x_sign, y_sign = sel("sign")
    x_exp, y_exp = sel("exp")
    x_sig, y_sig = sel("sig")

    eff_sub = (x_sign ^ y_sign) == 1
    shamt = m(x_exp - y_exp, 5)
    y_wide = m((y_sig << 30) >> shamt, 41)
    y_al = (m(y_wide >> 28, 13) << 1) | (m(y_wide, 28) != 0)
    x_al = x_sig << 3
    s = np.where(eff_sub, m(x_al - y_al, 15), m(x_al + y_al, 15))

    lz = np.full_like(s, 14)
    for i in range(14):
        lz = np.where(((s >> i) & 1) == 1, 13 - i, lz)
    lshift = np.where(lz < x_exp, lz, m(x_exp - 1, 5))
    norm_l = m(s << lshift, 15)

    carry = ((s >> 14) & 1) == 1
    n_exp = np.where(carry, m(x_exp + 1, 7), m(x_exp - lshift, 7))
    n_mant = np.where(carry, m(s >> 4, 11), m(norm_l >> 3, 11))
    n_g = np.where(carry, (s >> 3) & 1, (norm_l >> 2) & 1)
    n_s = np.where(carry, (s & 7) != 0, (norm_l & 3) != 0).astype(np.int64)

    rp, of, uf, nx = round_pack(x_sign, n_exp, n_mant, n_g, n_s)

    any_nan = A["snan"] | A["qnan"] | B["snan"] | B["qnan"]
    inf_inf = A["inf"] & B["inf"] & (A["sign"] != B["sign"])
    any_inf = A["inf"] | B["inf"]
    zero_sum = s == 0
    normal_path = ~any_nan & ~inf_inf & ~any_inf & ~zero_sum

    result = np.where(any_nan, QNAN,
             np.where(inf_inf, QNAN,
             np.where(any_inf, np.where(A["inf"], a, b_eff),
             np.where(zero_sum, (A["sign"] & B["sign"]) << 15, rp))))
    nv = np.where(any_nan, A["snan"] | B["snan"], inf_inf)
    flags = nv * 16 + (normal_path & of) * 4 + (normal_path & uf) * 2 + (normal_path & nx) * 1
    return result, flags.astype(np.int64)


def mul(a, b):
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    A, B = classify(a), classify(b)
    r_sign = A["sign"] ^ B["sign"]

    prod = A["sig"] * B["sig"]
    p = np.zeros_like(prod)
    for i in range(22):
        p = np.where(((prod >> i) & 1) == 1, i, p)

    e_sum = m(p + A["exp"] + B["exp"], 7)
    tiny = e_sum < 36
    r_exp = np.where(tiny, 1, m(e_sum - 35, 7))
    rsh = np.where(tiny, m(36 - m(e_sum, 6), 6), 0)

    lsh = m(21 - p, 5)
    prod_n = m(prod << lsh, 22)
    wide = m((prod_n.astype(np.uint64) << np.uint64(40)) >> rsh.astype(np.uint64), 62).astype(np.uint64)
    n_mant = ((wide >> np.uint64(51)) & np.uint64(0x7FF)).astype(np.int64)
    n_g = ((wide >> np.uint64(50)) & np.uint64(1)).astype(np.int64)
    n_s = ((wide & np.uint64((1 << 50) - 1)) != 0).astype(np.int64)

    rp, of, uf, nx = round_pack(r_sign, r_exp, n_mant, n_g, n_s)

    any_nan = A["snan"] | A["qnan"] | B["snan"] | B["qnan"]
    inf_zero = (A["inf"] & B["zero"]) | (A["zero"] & B["inf"])
    any_inf = A["inf"] | B["inf"]
    any_zero = A["zero"] | B["zero"]
    normal_path = ~any_nan & ~inf_zero & ~any_inf & ~any_zero

    result = np.where(any_nan, QNAN,
             np.where(inf_zero, QNAN,
             np.where(any_inf, (r_sign << 15) | 0x7C00,
             np.where(any_zero, r_sign << 15, rp))))
    nv = np.where(any_nan, A["snan"] | B["snan"], inf_zero)
    flags = nv * 16 + (normal_path & of) * 4 + (normal_path & uf) * 2 + (normal_path & nx) * 1
    return result, flags.astype(np.int64)


def fpu(a, b, op):
    if op == 0: return addsub(a, b, 0)
    if op == 1: return addsub(a, b, 1)
    if op == 2: return mul(a, b)
    n = np.asarray(a).shape
    return np.full(n, QNAN, dtype=np.int64), np.full(n, 16, dtype=np.int64)
