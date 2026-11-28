// -----------------------------------------------------------------------------
// fp16_mul.v
// IEEE 754 binary16 multiplier, round-to-nearest-even, full subnormal support
// (as inputs and as results), exception flags. Purely combinational.
//
// Datapath
//   1. Classify both operands, handle NaN / infinity / zero.
//   2. Multiply the 11-bit significands -> 22-bit exact product.
//   3. Find the leading 1 of the product (subnormal inputs can leave leading
//      zeros) and compute the result exponent.
//   4. Normalize the leading 1 to the top. If the exponent is below 1 the
//      result is subnormal: shift right instead and collect a sticky bit.
//   5. Round (RNE) and pack in fp16_round_pack.
//
// Exponent bookkeeping (all unsigned, no negative numbers in hardware):
//   a = sig_a x 2^(ea-25), b = sig_b x 2^(eb-25)   (sig = 11-bit integer, bias 15)
//   If the product's leading 1 is at bit p, the biased result exponent is
//       E = p + ea + eb - 35
//   e_sum = p + ea + eb, so  E >= 1  <=>  e_sum >= 36.
// -----------------------------------------------------------------------------
module fp16_mul (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg  [15:0] result,
    output reg         invalid,
    output reg         overflow,
    output reg         underflow,
    output reg         inexact
);
    localparam [15:0] QNAN = 16'h7E00;                // canonical quiet NaN

    // ---------------- 1. classify ----------------
    wire        a_sign, b_sign;
    wire [4:0]  a_exp,  b_exp;
    wire [10:0] a_sig,  b_sig;
    wire        a_snan, a_qnan, a_inf, a_zero, a_subn, a_norm;
    wire        b_snan, b_qnan, b_inf, b_zero, b_subn, b_norm;

    fp16_classify u_cls_a (.x(a), .sign(a_sign), .exp_eff(a_exp), .sig(a_sig),
                           .is_snan(a_snan), .is_qnan(a_qnan), .is_inf(a_inf),
                           .is_zero(a_zero), .is_subnormal(a_subn), .is_normal(a_norm));
    fp16_classify u_cls_b (.x(b), .sign(b_sign), .exp_eff(b_exp), .sig(b_sig),
                           .is_snan(b_snan), .is_qnan(b_qnan), .is_inf(b_inf),
                           .is_zero(b_zero), .is_subnormal(b_subn), .is_normal(b_norm));

    // IEEE 754: the sign of a product is the XOR of the operand signs
    wire r_sign = a_sign ^ b_sign;

    // ---------------- 2. multiply significands ----------------
    wire [21:0] prod = a_sig * b_sig;

    // ---------------- 3. leading-one position and exponent ----------------
    reg [4:0] p;
    integer   i;
    always @(*) begin
        p = 5'd0;
        for (i = 0; i <= 21; i = i + 1)
            if (prod[i]) p = i;
    end

    wire [6:0] e_sum = {2'b00, p} + {2'b00, a_exp} + {2'b00, b_exp};   // 2 .. 81
    wire       tiny  = (e_sum < 7'd36);                                // result exponent < 1
    wire [6:0] r_exp = tiny ? 7'd1 : (e_sum - 7'd35);
    wire [5:0] rsh   = tiny ? (6'd36 - e_sum[5:0]) : 6'd0;             // 0 .. 34

    // ---------------- 4. normalize / denormalize ----------------
    wire [4:0]  lsh    = 5'd21 - p;
    wire [21:0] prod_n = prod << lsh;                     // leading 1 now at bit 21
    // 22 product bits + 40 zeros: a right shift of up to 34 loses nothing
    wire [61:0] wide   = {prod_n, 40'd0} >> rsh;
    wire [10:0] n_mant = wide[61:51];
    wire        n_g    = wide[50];
    wire        n_s    = |wide[49:0];

    // ---------------- 5. round and pack ----------------
    wire [15:0] rp_result;
    wire        rp_of, rp_uf, rp_nx;

    fp16_round_pack u_rp (.sign(r_sign), .exp_in(r_exp), .mant_in(n_mant),
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
            result  = QNAN;
            invalid = a_snan | b_snan;
        end
        else if ((a_inf & b_zero) | (a_zero & b_inf)) begin
            // infinity x 0 has no meaningful value
            result  = QNAN;
            invalid = 1'b1;
        end
        else if (a_inf | b_inf) begin
            result  = {r_sign, 5'b11111, 10'd0};
        end
        else if (a_zero | b_zero) begin
            result  = {r_sign, 15'd0};
        end
        else begin
            overflow  = rp_of;
            underflow = rp_uf;
            inexact   = rp_nx;
        end
    end
endmodule
