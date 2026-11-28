// -----------------------------------------------------------------------------
// fp16_addsub.v
// IEEE 754 binary16 adder / subtractor, round-to-nearest-even, full subnormal
// support, exception flags. Purely combinational.
//
//   sub = 0 : result = a + b
//   sub = 1 : result = a - b   (computed as a + (-b))
//
// Datapath
//   1. Classify both operands, handle NaN / infinity.
//   2. Swap so that X is the operand with the larger magnitude.
//   3. Align: shift Y's significand right by (expX - expY), keeping a guard bit,
//      a round bit and a sticky bit (OR of everything shifted further out).
//   4. Add or subtract the significands (effective operation = signX ^ signY).
//   5. Normalize: carry-out -> shift right 1; leading zeros -> shift left, but
//      never below exponent 1 (that case produces a subnormal).
//   6. Round (RNE) and pack in fp16_round_pack.
// -----------------------------------------------------------------------------
module fp16_addsub (
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire        sub,
    output reg  [15:0] result,
    output reg         invalid,
    output reg         overflow,
    output reg         underflow,
    output reg         inexact
);
    localparam [15:0] QNAN = 16'h7E00;                // canonical quiet NaN

    // a - b == a + (-b): flip the sign of b for subtraction
    wire [15:0] b_eff = {b[15] ^ sub, b[14:0]};

    // ---------------- 1. classify ----------------
    wire        a_sign, b_sign;
    wire [4:0]  a_exp,  b_exp;
    wire [10:0] a_sig,  b_sig;
    wire        a_snan, a_qnan, a_inf, a_zero, a_subn, a_norm;
    wire        b_snan, b_qnan, b_inf, b_zero, b_subn, b_norm;

    fp16_classify u_cls_a (.x(a),     .sign(a_sign), .exp_eff(a_exp), .sig(a_sig),
                           .is_snan(a_snan), .is_qnan(a_qnan), .is_inf(a_inf),
                           .is_zero(a_zero), .is_subnormal(a_subn), .is_normal(a_norm));
    fp16_classify u_cls_b (.x(b_eff), .sign(b_sign), .exp_eff(b_exp), .sig(b_sig),
                           .is_snan(b_snan), .is_qnan(b_qnan), .is_inf(b_inf),
                           .is_zero(b_zero), .is_subnormal(b_subn), .is_normal(b_norm));

    // ---------------- 2. order by magnitude ----------------
    // For finite values, comparing {exponent, fraction} as an unsigned integer
    // compares magnitudes.
    wire        swap   = (b_eff[14:0] > a[14:0]);
    wire        x_sign = swap ? b_sign : a_sign;
    wire [4:0]  x_exp  = swap ? b_exp  : a_exp;
    wire [10:0] x_sig  = swap ? b_sig  : a_sig;
    wire        y_sign = swap ? a_sign : b_sign;
    wire [4:0]  y_exp  = swap ? a_exp  : b_exp;
    wire [10:0] y_sig  = swap ? a_sig  : b_sig;

    wire        eff_sub = x_sign ^ y_sign;
    wire [4:0]  shamt   = x_exp - y_exp;              // 0 .. 29, never negative

    // ---------------- 3. align ----------------
    // 11 significand bits + 30 zeros: a shift of up to 29 loses nothing,
    // so the sticky bit is computed exactly.
    wire [40:0] y_wide = {y_sig, 30'd0} >> shamt;
    wire [13:0] y_al   = {y_wide[40:28], |y_wide[27:0]};  // {sig[10:0], G, R, S}
    wire [13:0] x_al   = {x_sig, 3'b000};

    // ---------------- 4. add / subtract ----------------
    // |X| >= |Y| guarantees the subtraction never goes negative.
    wire [14:0] sum = eff_sub ? ({1'b0, x_al} - {1'b0, y_al})
                              : ({1'b0, x_al} + {1'b0, y_al});

    // ---------------- 5. normalize ----------------
    // Leading-zero count of sum[13:0] (14 when sum[13:0] is zero).
    reg [3:0] lz;
    integer   i;
    always @(*) begin
        lz = 4'd14;
        for (i = 0; i <= 13; i = i + 1)
            if (sum[i]) lz = 13 - i;
    end

    // Left shift is limited so the exponent stays >= 1 (gradual underflow).
    wire [4:0]  lz5    = {1'b0, lz};
    wire [4:0]  lshift = (lz5 < x_exp) ? lz5 : (x_exp - 5'd1);
    wire [14:0] norm_l = sum << lshift;

    wire        carry  = sum[14];
    wire [6:0]  n_exp  = carry ? ({2'b00, x_exp} + 7'd1) : ({2'b00, x_exp} - {2'b00, lshift});
    wire [10:0] n_mant = carry ? sum[14:4] : norm_l[13:3];
    wire        n_g    = carry ? sum[3]   : norm_l[2];
    wire        n_s    = carry ? (|sum[2:0]) : (|norm_l[1:0]);

    // ---------------- 6. round and pack ----------------
    wire [15:0] rp_result;
    wire        rp_of, rp_uf, rp_nx;

    fp16_round_pack u_rp (.sign(x_sign), .exp_in(n_exp), .mant_in(n_mant),
                          .guard(n_g), .sticky(n_s), .result(rp_result),
                          .overflow(rp_of), .underflow(rp_uf), .inexact(rp_nx));

    // ---------------- special cases and output select ----------------
    always @(*) begin
        result    = rp_result;
        invalid   = 1'b0;
        overflow  = 1'b0;
        underflow = 1'b0;
        inexact   = 1'b0;

        if (a_snan | a_qnan | b_snan | b_qnan) begin
            // any NaN in -> quiet NaN out; a signaling NaN also raises invalid
            result  = QNAN;
            invalid = a_snan | b_snan;
        end
        else if (a_inf & b_inf & (a_sign != b_sign)) begin
            // (+inf) + (-inf) has no meaningful value
            result  = QNAN;
            invalid = 1'b1;
        end
        else if (a_inf | b_inf) begin
            result  = a_inf ? a : b_eff;              // infinity keeps its (effective) sign
        end
        else if (sum == 15'd0) begin
            // exact zero: -0 only if both operands are negative (x - x = +0 in RNE)
            result  = {a_sign & b_sign, 15'd0};
        end
        else begin
            overflow  = rp_of;
            underflow = rp_uf;
            inexact   = rp_nx;
        end
    end
endmodule
