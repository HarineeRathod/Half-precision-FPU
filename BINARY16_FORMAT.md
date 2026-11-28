# The binary16 Number Format, Explained

This note explains the three ideas the FPU's datapath is built on:

1. **implicit normalization** (the hidden bit)
2. why **subnormals use a scale of 2⁻¹⁴**, not 2⁻¹⁵
3. how far apart neighbouring numbers are (the **step size**, or ULP)

Each section points to the RTL lines that implement it.

---

## 1. Bit layout

```
 15  14       10  9                 0
+---+-----------+--------------------+
| s |  exponent |      fraction      |
+---+-----------+--------------------+
  1      5 bits        10 bits
```

- **s**: sign (0 = positive, 1 = negative)
- **exponent (E)**: stored with a bias of 15, so the field value `E` means 2^(E − 15)
- **fraction (f)**: the 10 bits after the binary point

| Exponent field | Fraction | Class | Value |
|---|---|---|---|
| 0 | 0 | zero | ±0 |
| 0 | ≠ 0 | subnormal | ±**0**.f × 2⁻¹⁴ |
| 1 … 30 | any | normal | ±**1**.f × 2^(E − 15) |
| 31 | 0 | infinity | ±∞ |
| 31 | ≠ 0 | NaN | f[9] = 1: quiet, f[9] = 0: signaling |

---

## 2. Implicit normalization (the hidden bit)

A normalized binary number always starts with a 1 (`1.xxxx × 2ᵉ`). Because that 1 is always there, it doesn't need to be stored. Binary16 stores only the 10 fraction bits, and the leading 1 is *implied*.

```
stored:   0 01111 1000000000        (0x3E00)
meaning:  1.1000000000 × 2^(15-15) = 1.5
          ^
          hidden bit: never stored
```

This gives **11 bits of precision from a 10-bit field**.

The alternative, **explicit normalization**, stores the leading 1 as a real bit. The x87 80-bit extended format does this. It uses one more bit of storage, but the hardware never has to reconstruct the 1.

### Where the hidden bit appears in the RTL

| Stage | What happens | Code |
|---|---|---|
| Unpack | the hidden bit is added back: 1 for normals, 0 for zero and subnormals | `fp16_classify.v`: `assign sig = {~e_zero, f};` |
| Compute | the adder and multiplier work on the full 11-bit significand; the hidden bit is a real bit in the arithmetic | `fp16_addsub.v`, `fp16_mul.v` |
| Normalize | the result is shifted until its leading 1 is back at bit 10 | adder: shift right on carry-out, shift left by the leading-zero count; multiplier: shift the product's leading 1 to the top |
| Pack | the hidden bit is dropped; only bits [9:0] are stored | `fp16_round_pack.v`: `{sign, exp_fin[4:0], mant_fin[9:0]}` |

### Where implicit normalization stops

When a result is too small to have a leading 1 at the smallest exponent, it can't be normalized. It is stored as a **subnormal** with a hidden **0**. That's why:

- the adder never shifts the exponent below 1: `lshift = min(lz, x_exp − 1)`
- the multiplier shifts *right* for tiny results (`rsh` in `fp16_mul.v`)

---

## 3. Why subnormals use 2⁻¹⁴ and not 2⁻¹⁵

The usual rule "exponent field minus 15" would give 0 − 15 = −15 for subnormals. The standard uses **−14** instead, so the subnormals continue exactly where the normal numbers stop.

The smallest normal number has exponent field 1:

```
smallest normal = 1.0000000000 × 2^(1-15) = 2^-14
```

### With 2⁻¹⁴ (IEEE 754): no gap

```
largest subnormal = 0.1111111111 × 2^-14 = 2^-14 − 2^-24
smallest normal   = 1.0000000000 × 2^-14 = 2^-14
```

These two numbers are neighbours, exactly 2⁻²⁴ apart.

### With 2⁻¹⁵: a hole

```
largest subnormal ≈ 0.1111111111 × 2^-15 ≈ 2^-15
smallest normal   = 1.0000000000 × 2^-14 = 2^-14
```

Nothing could be represented between about 2⁻¹⁵ and 2⁻¹⁴. That missing range is half the distance down to zero.

### The rule

Exponent field 0 changes **the hidden bit** (1 → 0), not the scale. Subnormals share the scale of exponent field 1:

| Format | Bias | Subnormal scale |
|---|---|---|
| binary16 (half) | 15 | 2^(1 − 15) = **2⁻¹⁴** |
| binary32 (single) | 127 | 2^(1 − 127) = **2⁻¹²⁶** |
| binary64 (double) | 1023 | 2^(1 − 1023) = **2⁻¹⁰²²** |

In the RTL: `fp16_classify.v` reports exponent 1 for subnormals.

```verilog
assign exp_eff = e_zero ? 5'd1 : e;   // subnormals use the scale of exponent 1
assign sig     = {~e_zero, f};        // ... but with a hidden 0
```

---

## 4. Step size: how far apart neighbouring numbers are

### Where 2⁻²⁴ comes from

The 10 fraction bits sit after the binary point, and each position is worth half the one before:

```
 0 . f9    f8    f7   ...  f1    f0
     2^-1  2^-2  2^-3 ...  2^-9  2^-10
```

The smallest possible change is flipping the last bit, f0, which changes the fraction by 2⁻¹⁰. The fraction is then multiplied by the scale, so the smallest change is multiplied by it too:

```
step = (weight of the last fraction bit) × (scale)
     = 2^-10 × 2^-14
     = 2^(-10 - 14)        because 2^a × 2^b = 2^(a+b)
     = 2^-24
```

The step between neighbouring numbers is called one **ULP** (unit in the last place).

### Check with real bit patterns

**Subnormals** (exponent field 0, scale 2⁻¹⁴, hidden bit 0):

| Hex | Significand | Value |
|---|---|---|
| `0001` | 0.0000000001 | 2⁻¹⁰ × 2⁻¹⁴ = **1 × 2⁻²⁴** |
| `0002` | 0.0000000010 | 2⁻⁹ × 2⁻¹⁴ = **2 × 2⁻²⁴** |
| `0003` | 0.0000000011 | **3 × 2⁻²⁴** |
| `03FF` | 0.1111111111 | **1023 × 2⁻²⁴** (largest subnormal) |

**Normals with exponent field 1** (scale 2⁻¹⁴, hidden bit 1):

| Hex | Significand | Value |
|---|---|---|
| `0400` | 1.0000000000 | 2⁻¹⁴ = **1024 × 2⁻²⁴** (smallest normal) |
| `0401` | 1.0000000001 | (1 + 2⁻¹⁰) × 2⁻¹⁴ = **1025 × 2⁻²⁴** |
| `0402` | 1.0000000010 | **1026 × 2⁻²⁴** |

(2⁻¹⁴ = 1024 × 2⁻²⁴, because 2¹⁰ = 1024.)

### It is simply counting

In units of 2⁻²⁴, the numbers around the subnormal/normal boundary are:

```
0, 1, 2, 3, ... , 1022, 1023 | 1024, 1025, 1026, ...
        subnormals           |  normals (exponent field 1)
      0x0001 ... 0x03FF      |  0x0400 ...
```

1023 is followed directly by 1024: there is no gap.

*Decimal analogy:* with 3 digits after the decimal point and a scale of 10⁻⁵, the smallest step is 10⁻³ × 10⁻⁵ = 10⁻⁸. Binary16 works the same way, with 10 binary digits and a scale of 2⁻¹⁴.

### The step doubles with every exponent

The step is always 2⁻¹⁰ × scale, so each time the exponent goes up by 1, the spacing doubles:

| Exponent field | Range | Step (ULP) |
|---|---|---|
| 0 (subnormal) | 0 … 2⁻¹⁴ | 2⁻²⁴ ≈ 6.0 × 10⁻⁸ |
| 1 | 2⁻¹⁴ … 2⁻¹³ | 2⁻²⁴ |
| 2 | 2⁻¹³ … 2⁻¹² | 2⁻²³ |
| 15 | 1 … 2 | 2⁻¹⁰ ≈ 0.000977 |
| 25 | 1024 … 2048 | 1 |
| 26 | 2048 … 4096 | 2 |
| 30 | 32768 … 65504 | 32 |

So floating-point numbers are dense near zero and sparse near the maximum. For example, at 2048 the step is 2, so `2048 + 1` cannot be represented: it is exactly halfway between 2048 and 2050 and rounds to 2048, the even neighbour. This is directed test vector `6800 + 3C00` in `sim/vectors.hex`.

---

## 5. Why this matters: gradual underflow

Because there is no gap near zero, binary16 guarantees:

> **x − y = 0 if and only if x = y**

Any two different binary16 numbers differ by at least 2⁻²⁴, and every multiple of 2⁻²⁴ below 2⁻¹⁴ is representable as a subnormal. So the difference of two different numbers is never flushed to zero.

Without subnormals (or with a gap), two different numbers could subtract to 0, and code such as

```c
if (x != y) z = 1.0 / (x - y);
```

could still divide by zero. This is called **gradual underflow**. It's also why the FPU never needs to raise the underflow flag for addition or subtraction: a subnormal result of add/sub is always exact.

---

## 6. Quick reference

| Quantity | Hex | Value |
|---|---|---|
| smallest subnormal | `0001` | 2⁻²⁴ ≈ 5.96 × 10⁻⁸ |
| largest subnormal | `03FF` | 1023 × 2⁻²⁴ ≈ 6.098 × 10⁻⁵ |
| smallest normal | `0400` | 2⁻¹⁴ ≈ 6.104 × 10⁻⁵ |
| 1.0 | `3C00` | 1 |
| next number after 1.0 | `3C01` | 1 + 2⁻¹⁰ ≈ 1.000977 |
| largest finite | `7BFF` | 65504 |
| +∞ / −∞ | `7C00` / `FC00` | |
| canonical quiet NaN (FPU output) | `7E00` | |
