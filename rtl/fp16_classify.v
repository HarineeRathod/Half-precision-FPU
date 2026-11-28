// -----------------------------------------------------------------------------
// fp16_classify.v
// Decodes one IEEE 754-2019 binary16 (half-precision) value.
//
//   bit 15      : sign
//   bits 14..10 : biased exponent (bias = 15)
//   bits  9..0  : fraction
//
//   exponent | fraction | class
//   ---------+----------+-------------------------------------------
//   0        | 0        | zero        (+0 / -0)
//   0        | != 0     | subnormal   value = 0.f x 2^-14
//   1..30    | any      | normal      value = 1.f x 2^(exp-15)
//   31       | 0        | infinity
//   31       | != 0     | NaN  (f[9] = 1 -> quiet NaN, f[9] = 0 -> signaling NaN)
//
// This module replaces the "converter" module that the original project
// instantiated but never included.
// -----------------------------------------------------------------------------
module fp16_classify (
    input  wire [15:0] x,
    output wire        sign,
    output wire [4:0]  exp_eff,      // biased exponent; zero/subnormal report 1 (true scale 2^-14)
    output wire [10:0] sig,          // significand with hidden bit: 1.f (normal) or 0.f (zero/subnormal)
    output wire        is_snan,
    output wire        is_qnan,
    output wire        is_inf,
    output wire        is_zero,
    output wire        is_subnormal,
    output wire        is_normal
);
    wire [4:0] e      = x[14:10];
    wire [9:0] f      = x[9:0];
    wire       e_max  = &e;          // exponent all ones
    wire       e_zero = ~|e;         // exponent all zeros
    wire       f_zero = ~|f;         // fraction all zeros

    assign sign         = x[15];
    assign is_snan      = e_max  & ~f_zero & ~f[9];
    assign is_qnan      = e_max  &  f[9];
    assign is_inf       = e_max  &  f_zero;
    assign is_zero      = e_zero &  f_zero;
    assign is_subnormal = e_zero & ~f_zero;
    assign is_normal    = ~e_max & ~e_zero;

    assign exp_eff      = e_zero ? 5'd1 : e;
    assign sig          = {~e_zero, f};
endmodule
