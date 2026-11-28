// -----------------------------------------------------------------------------
// fpu.v  -  Half-precision (IEEE 754 binary16) floating-point unit, top level
//
// opcode | operation
// -------+-----------------------------------------------------------------
//   00   | out = a + b
//   01   | out = a - b
//   10   | out = a * b
//   11   | reserved for division (not implemented yet): returns quiet NaN
//        | and raises the invalid flag, so it can never be mistaken for a result
//
// Timing: inputs are sampled on the rising clock edge when in_valid = 1; the
// registered result appears one cycle later with out_valid = 1 (latency 1).
//
// out_flag : class of the result     {snan, qnan, infinity, zero, subnormal, normal}
//            (same bit order as the original project; exactly one bit is set.
//             snan is never set for a result: IEEE operations only output quiet NaNs)
// exc_flag : IEEE 754 exception flags {invalid, div_by_zero, overflow, underflow, inexact}
//            (div_by_zero is reserved for the future divider and is always 0)
//
// Rounding : round to nearest, ties to even (the IEEE 754 default)
// -----------------------------------------------------------------------------
module fpu (
    input  wire        clk,
    input  wire        rst_n,        // asynchronous, active-low reset
    input  wire        in_valid,
    input  wire [1:0]  opcode,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg         out_valid,
    output reg  [15:0] out,
    output reg  [5:0]  out_flag,
    output reg  [4:0]  exc_flag
);
    localparam [1:0] OP_ADD = 2'b00,
                     OP_SUB = 2'b01,
                     OP_MUL = 2'b10,
                     OP_DIV = 2'b11;
    localparam [15:0] QNAN  = 16'h7E00;

    // ---------------- execution units ----------------
    // One adder serves both add and sub (opcode[0] selects subtraction).
    wire [15:0] as_res;
    wire        as_nv, as_of, as_uf, as_nx;
    fp16_addsub u_addsub (.a(a), .b(b), .sub(opcode[0]), .result(as_res),
                          .invalid(as_nv), .overflow(as_of), .underflow(as_uf), .inexact(as_nx));

    wire [15:0] mul_res;
    wire        mul_nv, mul_of, mul_uf, mul_nx;
    fp16_mul    u_mul    (.a(a), .b(b), .result(mul_res),
                          .invalid(mul_nv), .overflow(mul_of), .underflow(mul_uf), .inexact(mul_nx));

    // ---------------- result select ----------------
    reg [15:0] res_d;
    reg [4:0]  exc_d;
    always @(*) begin
        case (opcode)
            OP_ADD, OP_SUB: begin res_d = as_res;  exc_d = {as_nv,  1'b0, as_of,  as_uf,  as_nx};  end
            OP_MUL:         begin res_d = mul_res; exc_d = {mul_nv, 1'b0, mul_of, mul_uf, mul_nx}; end
            default:        begin res_d = QNAN;    exc_d = 5'b10000;                               end
        endcase
    end

    // ---------------- result classification ----------------
    wire        r_sign;
    wire [4:0]  r_exp;
    wire [10:0] r_sig;
    wire        r_snan, r_qnan, r_inf, r_zero, r_subn, r_norm;
    fp16_classify u_cls_r (.x(res_d), .sign(r_sign), .exp_eff(r_exp), .sig(r_sig),
                           .is_snan(r_snan), .is_qnan(r_qnan), .is_inf(r_inf),
                           .is_zero(r_zero), .is_subnormal(r_subn), .is_normal(r_norm));

    // ---------------- output register ----------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out       <= 16'd0;
            out_flag  <= 6'd0;
            exc_flag  <= 5'd0;
        end
        else begin
            out_valid <= in_valid;
            if (in_valid) begin
                out      <= res_d;
                out_flag <= {r_snan, r_qnan, r_inf, r_zero, r_subn, r_norm};
                exc_flag <= exc_d;
            end
        end
    end
endmodule
