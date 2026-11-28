[README.md](https://github.com/user-attachments/files/35544638/README.md)
# Half-Precision Floating-Point Unit (IEEE 754 binary16)

A synthesizable Verilog FPU for 16-bit half-precision numbers that performs **addition, subtraction and multiplication** following IEEE 754-2019:

- round to nearest, ties to even (the IEEE default rounding mode)
- full **subnormal** support, as inputs and as results (gradual underflow)
- correct handling of **±0, ±∞, quiet NaN and signaling NaN**
- all five IEEE **exception flags** (invalid, divide-by-zero, overflow, underflow, inexact)
- guard, round and sticky bits, so every result is the correctly rounded value
- registered output with a valid handshake and an asynchronous active-low reset
- plain Verilog-2001: works with Quartus, Vivado, ModelSim/Questa, Xcelium and Icarus

Started as a B.Tech 6th-semester project (Nirma University) and later rewritten, with a golden model and an exhaustive verification flow.

## Interface

| Port        | Dir | Width | Description |
|-------------|-----|-------|-------------|
| `clk`       | in  | 1  | clock |
| `rst_n`     | in  | 1  | asynchronous reset, active low |
| `in_valid`  | in  | 1  | inputs are valid this cycle |
| `opcode`    | in  | 2  | `00` add, `01` sub, `10` mul, `11` reserved (division) |
| `a`, `b`    | in  | 16 | operands (binary16) |
| `out_valid` | out | 1  | `out` holds a new result (one cycle after `in_valid`) |
| `out`       | out | 16 | result (binary16) |
| `out_flag`  | out | 6  | class of the result `{snan, qnan, inf, zero, subnormal, normal}` |
| `exc_flag`  | out | 5  | exception flags `{invalid, div_by_zero, overflow, underflow, inexact}` |

- Latency: 1 clock cycle. A new operation can start every cycle.
- Opcode `11` is reserved for a future divider. Until then it returns a quiet NaN (`7E00`) and raises *invalid*, so it can never be mistaken for a real result.
- Every NaN result is the canonical quiet NaN `7E00`. A signaling NaN input raises *invalid*.
- Underflow is signaled when the result is tiny (below 2⁻¹⁴, detected before rounding) **and** inexact.

## Structure

```
rtl/
  fpu.v               top level: opcode decode, result classification, output register
  fp16_addsub.v       adder/subtractor: swap, align (G/R/S), add, normalize
  fp16_mul.v          multiplier: 11x11 significand product, normalize/denormalize
  fp16_round_pack.v   shared round-to-nearest-even, overflow/underflow, packing
  fp16_classify.v     operand decoder (zero / subnormal / normal / inf / NaN)
tb/
  tb_fpu.v            self-checking testbench (reads sim/vectors.hex)
model/
  fp16_ref.py         golden models: exact rational arithmetic, and a numpy version
  rtl_mirror.py       bit-exact Python mirror of the RTL datapath
  gen_vectors.py      generates sim/vectors.hex (directed + random vectors)
  exhaustive_check.py compares the mirror with the reference for all 2^32 input pairs
sim/
  vectors.hex         120,114 test vectors with expected results and flags
  run_icarus.sh       one-command simulation
fpu.qpf, fpu.qsf      Quartus project
docs/DESIGN_NOTES.md  how binary16 works and how each block is built
```

## Running the testbench

**Icarus Verilog** (Linux / macOS / Windows):

```sh
sh sim/run_icarus.sh          # add +vcd to dump fpu_tb.vcd for GTKWave
```

**ModelSim / Questa** (run from `sim/`):

```
vlog ../rtl/*.v ../tb/tb_fpu.v
vsim -c tb_fpu -do "run -all; quit"
```

**EDA Playground** (no install): paste `tb/tb_fpu.v` in the testbench pane and the five `rtl/` files in the design pane, upload `sim/vectors.hex`, choose Icarus Verilog 12 and run.

A passing run ends with:

```
RESULT: PASS - all checks matched the golden model
```

Regenerate the vectors (optional): `python3 model/gen_vectors.py [random_per_op] [seed]`.

## Verification

1. **Golden reference.** `fp16_ref.py` holds two independent references: one using exact rational arithmetic and a hand-written IEEE rounding routine, and one using exact float64 arithmetic followed by numpy's correctly-rounded conversion to float16. They agree on every vector tested.
2. **Exhaustive algorithm check.** `rtl_mirror.py` re-implements the RTL datapath bit for bit (same signal names and widths). It was compared against the reference for **all 2³² input pairs of add, sub and mul** (12.9 billion cases), checking the result bits and all exception flags: **0 mismatches**.
3. **RTL simulation.** `tb_fpu.v` checks the Verilog itself against 120,114 vectors (64 directed corner cases plus corner-biased and uniform random), comparing `out`, `exc_flag`, `out_flag` and the valid timing.

## What was fixed from the original version

| # | File | Problem | Effect |
|---|------|---------|--------|
| 1 | all | `converter` module was instantiated but never included | project did not compile |
| 2 | `fpu.v` | subtraction instance received the same inputs as addition | `a - b` returned `a + b` |
| 3 | `fpu.v` | `division` wire never driven; flags not updated for opcode 11 | output `X`, stale flags |
| 4 | `fpu.v` | blocking `=` in clocked block, no reset | simulation races, unknown start-up state |
| 5 | `sub_add.v` | `reg diff;` was 1 bit wide | exponent difference truncated: alignment wrong whenever exponents differ by more than 1 |
| 6 | `sub_add.v` | no mantissa written when the sum was already normalized (most common case) | stale/latched fraction bits |
| 7 | `sub_add.v` | carry and left-shift cases took off-by-one bit ranges | wrong fraction after normalization |
| 8 | `sub_add.v` | shift counter `sa` initialized once and never cleared | normalization amount grew between operations |
| 9 | `sub_add.v` | final zero check (`same exponent, same fraction, opposite sign`) overrode specials | `+inf + -inf` gave 0 instead of NaN |
| 10 | `sub_add.v` | zero check came before the NaN check; `+0 + -0` returned `-0` | wrong flags for `0 + NaN`; wrong zero sign |
| 11 | `sub_add.v` | `normal` / `subnormal` flags never set; no overflow detection | flags always 0; overflow wrapped |
| 12 | `mul.v` | overflow / underflow / subnormal paths commented out; 5-bit exponent sum wrapped | overflowing or underflowing products wrapped around instead of becoming ∞, a subnormal or 0 |
| 13 | `mul.v` | normalization only handled a product leading 1 at bit 21 or 20 | `subnormal * normal` gave wrong results |
| 14 | both | truncation instead of rounding, no guard/round/sticky | results off by up to 1 ULP |
| 15 | both | signaling NaN passed through unchanged, no invalid flag | not IEEE compliant |
| 16 | both | variables not assigned on every path of `always @(*)` | latches inferred in synthesis |

## Possible extensions

- division and square root (opcode `11` is already reserved)
- all five IEEE rounding modes (a 3-bit `rm` input, as in RISC-V)
- fused multiply-add
- pipelining the multiplier and adder for higher clock frequency
- a UVM or cocotb environment with functional coverage
