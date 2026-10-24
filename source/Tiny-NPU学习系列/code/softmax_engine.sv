// =============================================================================
// softmax_engine.sv - Full softmax over a row (3-pass architecture)
// Pass 1: Find max value (reduce_max for INT8, inline fp16_cmp_gt for FP16)
// Pass 2: Subtract max, exp via LUT, accumulate sum
// Pass 3: Normalize by reciprocal of sum, requantize to int8 / fp16
//
// Supports both INT8 (r_dtype==0) and FP16 (r_dtype==1) data paths.
// SRAM is 8-bit wide, so FP16 values require 2 reads/writes per element
// (low byte first, then high byte).
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;
import fp16_utils_pkg::*;


module softmax_engine (
    input logic clk,
    input logic rst_n,

    input  logic        cmd_valid,
    output logic        cmd_ready,
    input  logic [15:0] length,
    input  logic [15:0] src_base,
    input  logic [15:0] dst_base,
    input  logic [15:0] scale_factor,
    input  logic        causal_mask_en,
    input  logic [15:0] causal_limit,
    input  logic        cmd_dtype,       // 0: INT8, 1:FP16

    // SRAM read port
    output logic              sram_rd_en,
    output logic [      15:0] sram_rd_addr,
    input  logic [DATA_W-1:0] sram_rd_data,

    // SRAM write port
    output logic              sram_wr_en,
    output logic [      15:0] sram_wr_addr,
    output logic [DATA_W-1:0] sram_wr_data,

    // Scratch SRAM write port (registered intermediate EXP results))
    output logic        scratch_wr_en,
    output logic [15:0] scratch_wr_addr,
    output logic [15:0] scratch_wr_data,

    // Scratch SRAM read port
    output logic        scratch_rd_en,
    output logic [15:0] scratch_rd_addr,
    input  logic [15:0] scratch_rd_data,

    //  Status
    output logic done,
    output logic busy
);
    // FSM states
    typedef enum logic [4:0] {
        S_IDLE,
        // track the max data
        S_P1_READ,  // read data from sram
        S_P1_READ_HI,  // read high byte from sram used for fp16
        S_P1_FEED,  // track the max
        S_P1_WAIT,  // wait for the max_valid
        // compute the exp and sum, and write the exp to scratch sram
        S_P2_READ,  // read data from sram
        S_P2_READ_HI,  // read high byte from sram used for fp16
        S_P2_EXP,  // compute the exp (through LUT)
        S_P2_FEED,  // compute the sum and write to scratch
        S_P2_WAIT,  // wait for the sum_valid
        // read the exp from scratch sram, and compute the ssoftmax
        S_P3_RECIP,  // compute the 1/sum
        S_P3_RECIP_WAIT,  // wait for the 1/sum valid
        S_P3_READ,  // read data from scratch sram
        S_P3_NORM,  // normalize and write
        S_P3_WRITE_HI,  // write high byte of normalize
        S_DONE
    } state_t;

    state_t state, state_next;

    //registered command signal
    logic [15:0] r_length, r_src_base, r_dst_base, r_scale_factor, r_causal_limit;
    logic        r_causal_mask_en;
    logic [15:0] r_idx;
    logic        r_dtype;

    logic        is_fp16;
    assign is_fp16 = r_dtype;

    // FP16 internal registers
    logic        [15:0] r_max_val_fp16;  // FP16 max value in P1
    logic        [15:0] r_sum_val_fp16;  // FP16 sum value in P2
    logic        [15:0] r_recip_value_val_fp16;  // FP16 recip value in P3
    logic        [ 7:0] low_byte;  // The low byte of FP16
    logic        [15:0] fp16_rd_val;  // Assembled FP16 value from twic reads

    // Subtract module signal
    logic               rmax_start;
    logic               rmax_din_valid;
    logic signed [ 7:0] rmax_din;
    logic               rmax_din_last;
    logic               rmax_valid;
    logic signed [ 7:0] rmax_result;
    logic signed [ 7:0] r_max_val;

    // exp_LUT
    logic        [ 7:0] exp_addr;
    logic        [15:0] exp_data;

    // graph exp_LUT FP16
    logic        [ 7:0] exp_fp16_addr;
    logic        [15:0] exp_fp16_data;

    // Sum module
    logic               rsum_start;
    logic               rsum_din_valid;
    logic signed [15:0] rsum_din;
    logic               rsum_din_last;
    logic               rsum_valid;
    logic signed [31:0] rsum_result;
    logic        [15:0] r_sum_val;

    // recip lut
    logic        [ 7:0] recip_addr;
    logic        [15:0] recip_data;
    logic        [15:0] r_recip_val;

    //Pipeline registers
    logic signed [ 7:0] p_rd_data;
    logic        [15:0] p_exp_val;
    logic        [15:0] p_scratch;

    // FP16 intermediate computation wires
    logic        [15:0] fp16_diff;  // x-max
    logic        [15:0] fp16_norm_result;
    logic        [15:0] fp16_assembled;
    logic        [15:0] r_recip_val_fp16;

    // ----------------------------------------------------------------
    // Sub-module instantiations
    // ----------------------------------------------------------------
    reduce_max u_reduce_max (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (rmax_start),
        .din_valid (rmax_din_valid),
        .din       (rmax_din),
        .din_last  (rmax_din_last),
        .max_val   (rmax_result),
        .max_valid (rmax_valid)
    );

    exp_lut u_exp_lut (
        .clk      (clk),
        .addr     (exp_addr),
        .data_out (exp_data)
    );

    graph_exp_lut_fp16 u_exp_lut_fp16 (
        .clk      (clk),
        .addr     (exp_fp16_addr),
        .data_out (exp_fp16_data)
    );

    reduce_sum u_reduce_sum (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (rsum_start),
        .din_valid (rsum_din_valid),
        .din       (rsum_din),
        .din_last  (rsum_din_last),
        .sum_val   (rsum_result),
        .sum_valid (rsum_valid)
    );

    recip_lut u_recip_lut (
        .clk      (clk),
        .addr     (recip_addr),
        .data_out (recip_data)
    );

    // FSM transition
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end
        else begin
            state <= state_next;
        end
    end

    always_comb begin
        state_next = state;
        case (state)
            S_IDLE: if (cmd_valid) state_next = S_P1_READ;
            S_P1_READ: begin
                if (is_fp16) state_next = S_P1_READ_HI;
                else state_next = S_P1_FEED;
            end
            S_P1_READ_HI: state_next = S_P1_FEED;
            S_P1_FEED: begin
                if (r_idx == r_length - 1) begin
                    if (is_fp16) begin
                        state_next = S_P2_READ;
                    end
                    else begin
                        state_next = S_P1_WAIT;
                    end
                end
                else begin
                    state_next = S_P1_READ;
                end
            end
            S_P1_WAIT: if (rmax_valid) state_next = S_P2_READ;
            S_P2_READ: begin
                if (is_fp16) state_next = S_P2_READ_HI;
                else state_next = S_P2_EXP;
            end
            S_P2_READ_HI: state_next = S_P2_EXP;
            S_P2_EXP: state_next = S_P2_FEED;
            S_P2_FEED: begin
                if (r_idx == r_length - 1) begin
                    if (is_fp16) begin
                        state_next = S_P3_RECIP;
                    end
                    else begin
                        state_next = S_P2_WAIT;
                    end
                end
                else begin
                    state_next = S_P2_READ;
                end
            end
            S_P2_WAIT: if (rsum_valid) state_next = S_P3_RECIP;
            S_P3_RECIP: state_next = S_P3_RECIP_WAIT;
            S_P3_RECIP_WAIT: state_next = S_P3_READ;
            S_P3_READ: state_next = S_P3_NORM;
            S_P3_NORM: begin
                if (is_fp16) begin
                    state_next = S_P3_WRITE_HI;
                end
                else begin
                    if (r_idx == r_length - 1) begin
                        state_next = S_DONE;
                    end
                    else begin
                        state_next = S_P3_READ;
                    end
                end
            end
            S_P3_WRITE_HI: begin
                if (r_idx == r_length - 1) begin
                    state_next = S_DONE;
                end
                else begin
                    state_next = S_P3_READ;
                end
            end
            S_DONE: state_next = S_IDLE;
            default: state_next = S_IDLE;
        endcase
    end

    // Command Capture
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_length         <= '0;
            r_src_base       <= '0;
            r_dst_base       <= 0;
            r_scale_factor   <= 0;
            r_causal_limit   <= 0;
            r_causal_mask_en <= 0;
            r_dtype          <= '0;
        end
        else if (state == S_IDLE && cmd_valid) begin
            r_length         <= length;
            r_src_base       <= src_base;
            r_dst_base       <= dst_base;
            r_scale_factor   <= scale_factor;
            r_causal_limit   <= causal_limit;
            r_causal_mask_en <= causal_mask_en;
            r_dtype          <= cmd_dtype;
        end
    end

    // idx update
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_idx <= '0;
        end
        else if (state == S_IDLE) begin
            r_idx <= '0;
        end
        else if (state == S_P1_FEED && state_next == S_P1_READ) begin
            r_idx <= r_idx + 1'b1;  // Pass 1 -> next element
        end
        else if (!is_fp16 && state == S_P1_WAIT && rmax_valid) begin
            r_idx <= '0;  // Pass 1 done, reset for Pass 2
        end
        else if (is_fp16 && state == S_P1_FEED && r_idx == r_length - 1) begin
            r_idx <= '0;  // Pass 1 done, reset for Pass 2, for FP16
        end
        else if (state == S_P2_FEED && state_next == S_P2_READ) begin
            r_idx <= r_idx + 1'b1;  // Pass2 -> next element
        end
        else if (!is_fp16 && state == S_P2_WAIT && rsum_valid) begin
            r_idx <= '0;  // Pass 2 done, reset for Pass3
        end
        else if (is_fp16 && state == S_P2_FEED && r_idx == r_length - 1) begin
            r_idx <= '0;  // Pass 2 done, reset for Pass 3, for FP16
        end
        else if (!is_fp16 && state == S_P3_NORM && state_next == S_P3_READ) begin
            r_idx <= r_idx + 1'b1;  // Pass 3 -> next element write for INT8
        end
        else if (is_fp16 && state == S_P3_WRITE_HI && state_next == S_P3_READ) begin
            r_idx <= r_idx + 1'b1;  // Pass 3 -> next element write for FP16
        end
        else if (state == S_DONE) begin
            r_idx <= '0;  // Pass 3 done, reset
        end
    end

    // Latch max and sum value for INT8
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_max_val <= -8'sd128;
        end
        else if (rmax_valid) begin
            r_max_val <= rmax_result;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_sum_val <= 0;
        end
        else if (rsum_valid) begin
            r_sum_val <= rsum_result[15:0];
        end
    end

    // FP16 max track inline using fp16_cmp_gt
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_max_val_fp16 <= 16'hFC00;
        end
        else if (state == S_IDLE && cmd_valid) begin
            r_max_val_fp16 <= 16'hFC00;
        end
        else if (state == S_P1_FEED && is_fp16) begin
            // Use fp16_assembled (combinational) since fp16_rd_val hasn't
            // been registered yet at this point in the cycle
            if (r_causal_mask_en && r_idx > r_causal_limit) begin
                // Masked position: treat as -inf, don't update max
            end
            else if (fp16_cmp_gt(fp16_assembled, r_max_val_fp16)) begin
                r_max_val_fp16 <= fp16_assembled;
            end
        end
    end

    // FP16 sum accumulation inline using fp16_add
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_sum_val_fp16 <= '0;
        end
        else if (state == S_P1_FEED && is_fp16 && r_idx == r_length - 1) begin
            r_sum_val_fp16 <= '0;  // reset sum at the end of pass 1
        end
        else if (!is_fp16 && state == S_P1_WAIT && rmax_valid) begin
            r_sum_val_fp16 <= '0;
        end
        else if (state == S_P2_FEED && is_fp16) begin
            r_sum_val_fp16 <= fp16_add(r_sum_val_fp16, exp_fp16_data);
        end
    end

    // ----------------------------------------------------------------
    // FP16 reciprocal latch (pass 3)
    // Convert recip_lut output (Q0.16 unsigned) to FP16.
    // recip_lut[i] = round(65536 / i), so the value represents 1/sum_approx.
    // We convert this Q0.16 value to FP16 for use with fp16_mul.
    // ----------------------------------------------------------------
    // Q0.16 to FP16 conversion helper: value / 65536.0
    // The recip_lut output is an unsigned 16-bit integer representing a Q0.16
    // fractional value. To convert to FP16: treat it as an integer, convert
    // to FP16, then multiply by FP16(1/256) twice (or FP16(1/65536)).
    // Simpler: use the int-to-fp16 approach for the upper byte, then adjust exponent.
    //
    // Actually, simplest practical approach: we use the recip_lut output as-is
    // for INT8, and for FP16 we convert the 16-bit Q0.16 recip value to FP16
    // by finding the MSB and building the float.
    // ----------------------------------------------------------------
    function automatic logic [15:0] q016_to_fp16(input logic [15:0] val);
        // Convert unsigned Q0.16 fixed-point to FP16
        // Q0.16 means the value = val / 65536.0, range [0, 1)
        // Special case: val=0 -> +0.0
        logic   [15:0] result;
        logic   [ 4:0] msb_pos;
        logic   [ 4:0] fp16_exp;
        logic   [ 9:0] fp16_mant;
        integer        i;

        if (val == 16'h0000) begin
            result = 16'h0000;
            return result;
        end

        // Find MSB position (highest set bit, 0..15)
        msb_pos = 5'd0;
        for (i = 15; i >= 0; i = i - 1) begin
            if (val[i]) begin
                msb_pos = i[4:0];
                break;
            end
        end

        // The true value is val * 2^(-16)
        // If MSB is at position msb_pos, then val = 1.frac * 2^msb_pos
        // So true value = 1.frac * 2^(msb_pos - 16)
        // FP16 exponent = (msb_pos - 16) + 15 = msb_pos - 1
        // Since msb_pos is 0..15, fp16_exp ranges from -1 to 14
        // For msb_pos == 0: exponent = -1 -> underflow to zero (denormal)
        if (msb_pos == 5'd0) begin
            // Value = 1 * 2^(-16), too small for FP16 normals -> flush to zero
            result = 16'h0000;
            return result;
        end

        fp16_exp = msb_pos - 5'd1; // biased exponent (bias=15, unbiased = msb_pos-16, biased = msb_pos-16+15 = msb_pos-1)

        // Extract mantissa: remove implicit 1, left-align into 10 bits
        // The bits below msb_pos form the fraction
        case (msb_pos)
            5'd1:  fp16_mant = {val[0],    9'h000};
            5'd2:  fp16_mant = {val[1:0],  8'h00};
            5'd3:  fp16_mant = {val[2:0],  7'h00};
            5'd4:  fp16_mant = {val[3:0],  6'h00};
            5'd5:  fp16_mant = {val[4:0],  5'h00};
            5'd6:  fp16_mant = {val[5:0],  4'h0};
            5'd7:  fp16_mant = {val[6:0],  3'h0};
            5'd8:  fp16_mant = {val[7:0],  2'h0};
            5'd9:  fp16_mant = {val[8:0],  1'h0};
            5'd10: fp16_mant = val[9:0];
            5'd11: fp16_mant = val[10:1];
            5'd12: fp16_mant = val[11:2];
            5'd13: fp16_mant = val[12:3];
            5'd14: fp16_mant = val[13:4];
            5'd15: fp16_mant = val[14:5];
            default: fp16_mant = 10'h000;
        endcase

        result = {1'b0, fp16_exp, fp16_mant};  // always positive
        return result;
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_recip_val_fp16 <= '0;
        end
        else if (state == S_P3_RECIP_WAIT && is_fp16) begin
            r_recip_val_fp16 <= q016_to_fp16(recip_data);
        end
    end

    // SRAM read port
    always_comb begin
        sram_rd_en   = 0;
        sram_rd_addr = '0;
        case (state)
            S_P1_READ: begin
                sram_rd_en = 1'b1;
                if (is_fp16) sram_rd_addr = r_src_base + (r_idx << 1);
                else sram_rd_addr = r_src_base + r_idx;
            end
            S_P1_READ_HI: begin
                sram_rd_en   = 1'b1;
                sram_rd_addr = r_src_base + (r_idx << 1) + 1;
            end
            S_P2_READ: begin
                sram_rd_en = 1'b1;
                if (is_fp16) sram_rd_addr = r_src_base + (r_idx << 1);
                else sram_rd_addr = r_src_base + r_idx;
            end
            S_P2_READ_HI: begin
                sram_rd_en   = 1'b1;
                sram_rd_addr = r_src_base + (r_idx << 1) + 1;
            end
            default: begin
                sram_rd_en   = 0;
                sram_rd_addr = '0;
            end
        endcase
    end

    // contact the FP16
    always_ff @(posedge clk) begin
        if (state == S_P1_READ_HI || state == S_P2_READ_HI) low_byte <= sram_rd_data;
    end
    assign fp16_assembled = {sram_rd_data, low_byte};

    // assign for max tracking module signal (INT8) Pass 1
    assign rmax_start     = (state == S_IDLE && cmd_valid && cmd_dtype == 0);
    assign rmax_din_valid = (state == S_P1_FEED && !is_fp16);
    assign rmax_din_last  = (!is_fp16 && state == S_P1_FEED && r_idx == r_length - 1);

    logic signed [7:0] masked_input;
    always_comb begin
        if (r_causal_mask_en && r_idx > r_causal_limit) masked_input = -8'sd128;
        else masked_input = signed'(sram_rd_data);
    end

    assign rmax_din = masked_input;

    // exp LUT (INT8) Pass 2 for exp LUT
    logic signed [15:0] scaled_diff;
    logic signed [ 7:0] exp_input;

    always_comb begin
        scaled_diff = 16'(signed'(masked_input)) - 16'(signed'(r_max_val));
        if (scaled_diff < -16'sd128) exp_input = -8'sd128;
        else if (scaled_diff > 16'sd127) exp_input = 8'sd127;
        else exp_input = scaled_diff[7:0];
    end
    assign exp_addr = (state == S_P2_EXP && !is_fp16) ? exp_input : '0;

    // exp LUT (FP16) PASS 2
    logic [15:0] fp16_masked_input;
    always_comb begin
        if (r_causal_mask_en && r_idx > r_causal_limit) fp16_masked_input = 16'hFC00;
        else fp16_masked_input = fp16_assembled;
    end

    always_comb begin
        if (is_fp16) fp16_diff = fp16_sub(fp16_masked_input, r_max_val_fp16);
        else fp16_diff = 16'h0000;
    end
    assign exp_fp16_addr = (state == S_P2_EXP && is_fp16) ? fp16_diff[15:8] : '0;

    // Register the assembled value for use in subsequent cycles
    always_ff @(posedge clk) begin
        if (is_fp16 && (state == S_P1_FEED || state == S_P2_EXP)) fp16_rd_val <= fp16_assembled;
    end

    // assign for sum accumulation module signal (INT8) Pass 2 for sum accumulation
    assign rsum_start     = (state == S_P1_WAIT && rmax_valid && !is_fp16);
    assign rsum_din_valid = (state == S_P2_FEED && !is_fp16);
    assign rsum_din_last  = (state == S_P2_FEED && !is_fp16 && r_idx == r_length - 1);
    assign rsum_din       = 16'(signed'(exp_data));

    // write exp results to scratch sram while sum accumulation
    always_comb begin
        scratch_wr_en   = (state == S_P2_FEED);
        scratch_wr_addr = r_idx;
        if (is_fp16) scratch_wr_data = exp_fp16_data;
        else scratch_wr_data = exp_data;
    end

    // Pass 3 Normalizaiton

    // recip LUT
    always_comb begin
        if (is_fp16) recip_addr = r_sum_val_fp16[15:8];
        else recip_addr = r_sum_val[15:8];
    end

    always_ff @(posedge clk) begin
        if (state == S_P3_RECIP_WAIT && !is_fp16) r_recip_val <= recip_data;
    end

    // Scratch sram read port
    assign scratch_rd_en   = (state == S_P3_READ);
    assign scratch_rd_addr = r_idx;

    //INT8 normalization
    logic        [31:0] norm_product;
    logic        [14:0] norm_trunc;
    logic signed [ 7:0] norm_result;

    always_comb begin
        norm_product = 32'(scratch_rd_data) * 32'(r_recip_val);
        norm_trunc   = norm_product[31:17];
        if (norm_trunc > 15'd127) norm_result = 8'd127;
        else norm_result = norm_trunc[7:0];
    end

    // FP16 normalization
    assign fp16_norm_result = fp16_mul(scratch_rd_data, r_recip_val_fp16);

    // Latch fp16_normal_result
    logic [15:0] r_fp16_norm_out;
    always_ff @(posedge clk) begin
        if (state == S_P3_NORM && is_fp16) r_fp16_norm_out <= fp16_norm_result;
    end

    // result sram write port
    always_comb begin
        sram_wr_en   = 0;
        sram_wr_addr = '0;
        sram_wr_data = '0;
        case (state)
            S_P3_NORM: begin
                sram_wr_en = 1'b1;
                if (!is_fp16) begin
                    sram_wr_addr = r_dst_base + r_idx;
                    sram_wr_data = norm_result;
                end
                else begin
                    sram_wr_addr = r_dst_base + (r_idx << 1);
                    sram_wr_data = fp16_norm_result[7:0];
                end
            end
            S_P3_WRITE_HI: begin
                sram_wr_en   = 1'b1;
                sram_wr_addr = r_dst_base + (r_idx << 1) + 1;
                sram_wr_data = r_fp16_norm_out[15:8];
            end
            default: begin
                sram_wr_en   = 0;
                sram_wr_addr = '0;
                sram_wr_data = '0;
            end
        endcase
    end


    //Status output
    assign cmd_ready = (state == S_IDLE);
    assign done      = (state == S_DONE);
    assign busy      = (state != S_IDLE);

endmodule
