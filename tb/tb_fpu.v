// -----------------------------------------------------------------------------
// tb_fpu.v  -  self-checking testbench for fpu.v
//
// Reads sim/vectors.hex (made by model/gen_vectors.py from an exact-arithmetic
// golden model), applies one vector per clock and checks:
//   out       against the expected result (bit-exact)
//   exc_flag  against the expected IEEE exception flags
//   out_flag  against the class of the expected result
//   out_valid timing (latency 1)
// Prints the first mismatches in detail and a PASS/FAIL summary.
//
// Run from the sim/ directory, e.g. with Icarus Verilog:
//   iverilog -g2005 -o fpu_tb ../tb/tb_fpu.v ../rtl/*.v && vvp fpu_tb
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module tb_fpu;
    parameter MAXV       = 200000;
    parameter MAX_REPORT = 20;
    parameter VEC_FILE   = "vectors.hex";

    reg         clk = 1'b0;
    reg         rst_n = 1'b0;
    reg         in_valid = 1'b0;
    reg  [1:0]  opcode = 2'b00;
    reg  [15:0] a = 16'd0, b = 16'd0;
    wire        out_valid;
    wire [15:0] out;
    wire [5:0]  out_flag;
    wire [4:0]  exc_flag;

    fpu dut (.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .opcode(opcode),
             .a(a), .b(b), .out_valid(out_valid), .out(out),
             .out_flag(out_flag), .exc_flag(exc_flag));

    always #5 clk = ~clk;

    reg [63:0] vec [0:MAXV-1];
    integer    i, n_vec, errors, n_add, n_sub, n_mul, n_div;
    reg [15:0] exp_res;
    reg [4:0]  exp_exc;
    reg [5:0]  exp_cls;

    // class of a binary16 value: {snan, qnan, inf, zero, subnormal, normal}
    function [5:0] fp16_class;
        input [15:0] x;
        begin
            if (x[14:10] == 5'h1F)
                fp16_class = (x[9:0] == 10'd0) ? 6'b001000 : (x[9] ? 6'b010000 : 6'b100000);
            else if (x[14:10] == 5'h00)
                fp16_class = (x[9:0] == 10'd0) ? 6'b000100 : 6'b000010;
            else
                fp16_class = 6'b000001;
        end
    endfunction

    initial begin
        if ($test$plusargs("vcd")) begin
            $dumpfile("fpu_tb.vcd");
            $dumpvars(0, tb_fpu);
        end

        $readmemh(VEC_FILE, vec);
        n_vec = 0;
        while (n_vec < MAXV && vec[n_vec][63:56] !== 8'hFF) n_vec = n_vec + 1;
        if (n_vec == 0 || n_vec == MAXV) begin
            $display("ERROR: could not read %s (or no end-of-vectors line)", VEC_FILE);
            $finish;
        end
        $display("Loaded %0d vectors from %s", n_vec, VEC_FILE);

        errors = 0; n_add = 0; n_sub = 0; n_mul = 0; n_div = 0;

        // reset
        repeat (2) @(negedge clk);
        if (out_valid !== 1'b0 || out !== 16'd0) begin
            $display("ERROR: outputs not cleared by reset");
            errors = errors + 1;
        end
        rst_n = 1'b1;

        for (i = 0; i < n_vec; i = i + 1) begin
            @(negedge clk);
            opcode   = vec[i][57:56];
            a        = vec[i][55:40];
            b        = vec[i][39:24];
            in_valid = 1'b1;
            exp_res  = vec[i][23:8];
            exp_exc  = vec[i][4:0];
            exp_cls  = fp16_class(exp_res);
            case (opcode) 2'b00: n_add = n_add + 1; 2'b01: n_sub = n_sub + 1;
                          2'b10: n_mul = n_mul + 1; default: n_div = n_div + 1; endcase

            @(posedge clk); #1;
            if (out_valid !== 1'b1 || out !== exp_res || exc_flag !== exp_exc || out_flag !== exp_cls) begin
                errors = errors + 1;
                if (errors <= MAX_REPORT)
                    $display("MISMATCH #%0d  op=%b a=%h b=%h | out=%h exp=%h | exc=%b exp=%b | class=%b exp=%b | valid=%b",
                             i, opcode, a, b, out, exp_res, exc_flag, exp_exc, out_flag, exp_cls, out_valid);
            end
        end

        // out_valid must drop when in_valid drops
        @(negedge clk); in_valid = 1'b0;
        @(posedge clk); #1;
        if (out_valid !== 1'b0) begin
            $display("ERROR: out_valid stayed high without in_valid");
            errors = errors + 1;
        end

        $display("------------------------------------------------------------");
        $display("Vectors: %0d  (add %0d, sub %0d, mul %0d, reserved %0d)", n_vec, n_add, n_sub, n_mul, n_div);
        if (errors == 0) $display("RESULT: PASS - all checks matched the golden model");
        else             $display("RESULT: FAIL - %0d mismatches", errors);
        $display("------------------------------------------------------------");
        $finish;
    end
endmodule
