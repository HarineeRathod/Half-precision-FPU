Half-Precision Floating-Point Unit (FP16) in Verilog

A 16-bit floating-point unit for the IEEE 754 half-precision (binary16) format. It was written in Verilog as my B.Tech 6th-semester project (Electronics and Communication Engineering, Nirma University, 2023).

Scope: RTL design and simulation only. The design was not implemented or tested on FPGA hardware.

Supported operations
opcode	Operation	Module
2'b00	Addition	sub_add.v
2'b01	Subtraction	sub_add.v
2'b10	Multiplication	mul.v
2'b11	Reserved (division not implemented)	—
Top-level interface (fpu.v)
Port	Dir	Width	Description
a, b	in	16	FP16 operands
opcode	in	2	Operation select
clk	in	1	Result is registered on the rising edge
out	out	16	FP16 result
out_flag	out	6	{sNaN, qNaN, Infinity, Zero, Subnormal, Normal}
Design overview
Operand classification: each operand is decoded into sign, exponent and significand and classified as sNaN, qNaN, infinity, zero, subnormal or normal (converter module).
Addition/subtraction (sub_add.v):
Special cases come first: zero operands, NaN propagation, and infinity handling (∞ − ∞ gives a quiet NaN).
For finite operands, the smaller operand's significand is shifted to align exponents, then added or subtracted according to the signs.
The result is normalized: a carry-out shifts it right; leading zeros are found by a binary search and shifted out.
Equal-magnitude, opposite-sign operands give zero.
Multiplication (mul.v):
The result sign is the XOR of the operand signs.
NaNs are propagated, ∞ × 0 gives a quiet NaN, and ∞ × finite gives infinity.
Otherwise the 11×11-bit significands are multiplied, the exponents are added, and the product is normalized.
Files
FPU_/fpu.v        Top level: operation select, registered output and flags
FPU_/sub_add.v    FP16 adder/subtractor
FPU_/mul.v        FP16 multiplier
FPU_/converter.v  Operand decode and classification
fpu.qpf           Quartus II 14.0 project file (April 2023)
Verification

The design was simulated and checked by inspecting waveforms for representative normal, zero, infinity and NaN inputs. There is no self-checking testbench in this repository.

Known limitations
No rounding: results are truncated.
Division is not implemented (opcode = 2'b11).
The multiplier does not detect overflow, underflow or subnormal results (that logic is commented out).
A product of two subnormal operands is flushed to zero.
