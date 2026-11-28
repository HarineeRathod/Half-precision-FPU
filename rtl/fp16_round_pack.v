// -----------------------------------------------------------------------------
// fp16_round_pack.v
// Rounds a normalized significand to 11 bits with round-to-nearest, ties-to-even
// (the IEEE 754 default rounding mode) and packs the binary16 result.
// Shared by the adder/subtractor and the multiplier.
//
// Input value = mant_in[10].mant_in[9:0] x 2^(exp_in - 15), plus the bits below
// the LSB, summarized as:
//   guard  : the first bit below the LSB (worth exactly half an ULP)
//   sticky : OR of every bit below the guard bit
//
// Rounding decision (RNE):
//   guard = 0                 -> below half an ULP, truncate
//   guard = 1, sticky = 1     -> above half an ULP, round up
//   guard = 1, sticky = 0     -> exactly half (a tie): round up only if LSB is 1,
//                                so the result's LSB becomes 0 ("even")
//
// Callers guarantee exp_in >= 1. If mant_in[10] = 0 the value is subnormal and
// exp_in must be 1.
// -----------------------------------------------------------------------------
module fp16_round_pack (
    input  wire        sign,
    input  wire [6:0]  exp_in,
    input  wire [10:0] mant_in,
    input  wire        guard,
    input  wire        sticky,
    output reg  [15:0] result,
    output wire        overflow,
    output wire        underflow,
    output wire        inexact
);
    wire        round_up = guard & (sticky | mant_in[0]);
    wire [11:0] mant_rnd = {1'b0, mant_in} + {11'd0, round_up};
    wire        carry    = mant_rnd[11];                   // 1.11..1 + ULP = 10.00..0
    wire [10:0] mant_fin = carry ? mant_rnd[11:1] : mant_rnd[10:0];
    wire [7:0]  exp_fin  = {1'b0, exp_in} + {7'd0, carry};

    wire        lost     = guard | sticky;                 // some nonzero bits were discarded
    wire        tiny     = ~mant_in[10];                   // below 2^-14 before rounding

    assign overflow  = (exp_fin >= 8'd31);                 // too large for binary16
    assign inexact   = lost | overflow;
    assign underflow = tiny & lost;                        // tiny AND inexact (IEEE default)

    always @(*) begin
        if (overflow)
            result = {sign, 5'b11111, 10'd0};              // RNE overflows to infinity
        else if (mant_fin[10])
            result = {sign, exp_fin[4:0], mant_fin[9:0]};  // normal (a subnormal may round up into here)
        else
            result = {sign, 5'b00000, mant_fin[9:0]};      // subnormal, or zero after underflow
    end
endmodule
