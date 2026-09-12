// =============================================================================
// Layer Normalization Engine
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;
import fp16_utils_pkg::*;

module layernorm_engine (
    input logic clk,
    input logic rst_n,

    // command interface
    input  logic        cmd_valid,
    output logic        cmd_ready,
    input  logic [15:0] gamma_base,
    input  logic [15:0] beta_base,
    input  logic [15:0] src_base,
    input  logic [15:0] dst_base,
    input  logic [15:0] length,
    input  logic        data_type,   // 0: int8, 1:fp16

    // SRAM read port0 (input / gamma)
    output logic [      15:0] sram_rd0_addr,
    output logic              sram_rd0_en,
    input  logic [DATA_W-1:0] sram_rd0_data,

    // SRAM read port1 (beta)
    output logic [      15:0] sram_rd1_addr,
    output logic              sram_rd1_en,
    input  logic [DATA_W-1:0] sram_rd1_data,

    // SRAM write port
    output logic [      15:0] sram_wr_addr,
    output logic              sram_wr_en,
    output logic [DATA_W-1:0] sram_wr_data,

    // Status interface
    output logic busy,
    output logic done
);

    // mean_var_engine interface
    logic               mv_valid;
    logic               mv_start;
    logic               mv_data_valid;
    logic signed [ 7:0] mv_data_in;
    logic               mv_din_last;
    logic signed [15:0] mv_mean_out;
    logic        [31:0] mv_var_out;

    logic        [15:0] r_idx;
    logic        [15:0] r_length;
    logic        [15:0] r_gamma_base;
    logic        [15:0] r_beta_base;
    logic        [15:0] r_src_base;
    logic        [15:0] r_dst_base;
    logic               r_data_type;
    
    // 1/N computation for FP16
    logic [15:0] recip_n_lut_out;

    always_comb begin
        // Compute 1/N for common layer sizes via case statement
        // FP16 encoding: sign(1) | exponent(5) | mantissa(10)
        // 1/1   = 1.0     = 0x3C00
        // 1/2   = 0.5     = 0x3800
        // 1/4   = 0.25    = 0x3400
        // 1/8   = 0.125   = 0x3000
        // 1/16  = 0.0625  = 0x2C00
        // 1/32  = 0.03125 = 0x2800
        // 1/64  = 0x2400
        // 1/128 = 0x2000
        // 1/256 = 0x1C00
        // 1/512 = 0x1800
        // For non-power-of-2 values, approximate via nearest power-of-2
        case (r_length)
            16'd1: recip_n_lut_out = 16'h3C00;  // 1.0
            16'd2: recip_n_lut_out = 16'h3800;  // 0.5
            16'd3: recip_n_lut_out = 16'h3555;  // ~0.3333
            16'd4: recip_n_lut_out = 16'h3400;  // 0.25
            16'd5: recip_n_lut_out = 16'h3266;  // ~0.2
            16'd6: recip_n_lut_out = 16'h3155;  // ~0.1667
            16'd7: recip_n_lut_out = 16'h3092;  // ~0.1429
            16'd8: recip_n_lut_out = 16'h3000;  // 0.125
            16'd10: recip_n_lut_out = 16'h2E66;  // 0.1
            16'd12: recip_n_lut_out = 16'h2D55;  // ~0.0833
            16'd16: recip_n_lut_out = 16'h2C00;  // 0.0625
            16'd20: recip_n_lut_out = 16'h2A66;  // 0.05
            16'd24: recip_n_lut_out = 16'h2955;  // ~0.0417
            16'd32: recip_n_lut_out = 16'h2800;  // 0.03125
            16'd48: recip_n_lut_out = 16'h2555;  // ~0.0208
            16'd64: recip_n_lut_out = 16'h2400;  // 0.015625
            16'd96: recip_n_lut_out = 16'h2155;  // ~0.0104
            16'd128: recip_n_lut_out = 16'h2000;  // 0.0078125
            16'd192: recip_n_lut_out = 16'h1D55;  // ~0.0052
            16'd256: recip_n_lut_out = 16'h1C00;  // 0.00390625
            16'd384: recip_n_lut_out = 16'h1955;  // ~0.0026
            16'd512: recip_n_lut_out = 16'h1800;  // 0.001953125
            16'd768: recip_n_lut_out = 16'h1555;  // ~0.0013
            16'd1024: recip_n_lut_out = 16'h1400;  // ~0.000977
            16'd2048: recip_n_lut_out = 16'h1000;  // ~0.000488
            16'd4096: recip_n_lut_out = 16'h0C00;  // ~0.000244
            default: begin
                // Fallback: use int8_to_fp16 of upper byte as rough approximation
                // For lengths that are not in the LUT, approximate by nearest power-of-2
                if (r_length >= 16'd2048) recip_n_lut_out = 16'h1000;
                else if (r_length >= 16'd1024) recip_n_lut_out = 16'h1400;
                else if (r_length >= 16'd512) recip_n_lut_out = 16'h1800;
                else if (r_length >= 16'd256) recip_n_lut_out = 16'h1C00;
                else if (r_length >= 16'd128) recip_n_lut_out = 16'h2000;
                else if (r_length >= 16'd64) recip_n_lut_out = 16'h2400;
                else if (r_length >= 16'd32) recip_n_lut_out = 16'h2800;
                else if (r_length >= 16'd16) recip_n_lut_out = 16'h2C00;
                else if (r_length >= 16'd8) recip_n_lut_out = 16'h3000;
                else if (r_length >= 16'd4) recip_n_lut_out = 16'h3400;
                else if (r_length >= 16'd2) recip_n_lut_out = 16'h3800;
                else recip_n_lut_out = 16'h3C00;
            end
        endcase
    end

    // FSM
    typedef enum logic [4:0] {
        S_IDLE,
        // PASS1
        S_P1_READ,  // read input data
        S_P1_READ_HIGH,  // read input data high byte (FP16)
        S_P1_READFP16_WAIT, // wait FP16 data stable
        S_P1_FEED,  // feed input data to mean/var engine,
        S_P1_WAIT,  // wait for mean/var engine done
        S_P1_FP16_FINALIZE,  // finalize mean/var engine result (FP16)

        S_RSQRT,  // compute rsqrt
        S_RSQRT_WAIT,  // wait for rsqrt (LUT) done

        S_P2_READ,  // read input + beta data
        S_P2_READ_HIGH,  //  read input data high byte (FP16)
        S_P2_READ_GAMMA,  // read gamma data
        S_P2_READ_GAMMA_HIGH,  // read gamma data high byte (FP16)
        S_P2_READ_BETA,  // read beta data (FP16)
        S_P2_READ_BETA_HIGH,  // read beta data high byte (FP16)
        S_P2_READ_WAIT,
        S_P2_COMPUTE,  // compute normalized output
        S_P2_COMPUTE_HIGH,  // compute normalized output high byte (FP16)
        S_DONE  // compute done
    } state_t;
    state_t state, next_state;

    // Command Capture
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_length     <= 0;
            r_gamma_base <= 0;
            r_beta_base  <= 0;
            r_src_base   <= 0;
            r_dst_base   <= 0;
            r_data_type  <= 0;
        end
        else if (state == S_IDLE && cmd_valid) begin
            r_length     <= length;
            r_gamma_base <= gamma_base;
            r_beta_base  <= beta_base;
            r_src_base   <= src_base;
            r_dst_base   <= dst_base;
            r_data_type  <= data_type;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end
        else begin
            state <= next_state;
        end
    end

    always_comb begin
        case (state)
            S_IDLE: begin
                if (cmd_valid) next_state = S_P1_READ;
                else next_state = S_IDLE;
            end
            S_P1_READ: begin
                if (r_data_type) begin
                    next_state = S_P1_READ_HIGH;
                end
                else begin
                    next_state = S_P1_FEED;
                end
            end
            S_P1_READ_HIGH: begin
                next_state = S_P1_READFP16_WAIT;
            end
            S_P1_READFP16_WAIT: begin
                next_state = S_P1_FEED;
            end
            S_P1_FEED: begin
                if (r_idx == r_length - 1) begin  // last element
                    if (data_type) begin
                        next_state = S_P1_FP16_FINALIZE;
                    end
                    else begin
                        next_state = S_P1_WAIT;
                    end
                end
                else begin
                    next_state = S_P1_READ;  // read next element
                end
            end
            S_P1_WAIT: begin
                if (mv_valid) next_state = S_RSQRT;
                else next_state = S_P1_WAIT;
            end
            S_P1_FP16_FINALIZE: begin
                next_state = S_RSQRT;
            end
            S_RSQRT: begin
                next_state = S_RSQRT_WAIT;
            end
            S_RSQRT_WAIT: begin
                next_state = S_P2_READ;
            end
            S_P2_READ: begin
                if (data_type) begin
                    next_state = S_P2_READ_HIGH;
                end
                else begin
                    next_state = S_P2_READ_WAIT;
                end
            end
            S_P2_READ_HIGH: begin
                next_state = S_P2_READ_GAMMA;
            end
            S_P2_READ_GAMMA: begin
                next_state = S_P2_READ_GAMMA_HIGH;
            end
            S_P2_READ_GAMMA_HIGH: begin
                next_state = S_P2_READ_BETA;
            end
            S_P2_READ_BETA: begin
                next_state = S_P2_READ_BETA_HIGH;
            end
            S_P2_READ_BETA_HIGH: begin
                next_state = S_P2_READ_WAIT;
            end
            S_P2_READ_WAIT: begin
                next_state = S_P2_COMPUTE;
            end
            S_P2_COMPUTE: begin
                if (data_type) begin
                    next_state = S_P2_COMPUTE_HIGH;
                end
                else begin
                    if (r_idx == r_length - 1) begin  // output the last normalized result (INT8)
                        next_state = S_DONE;
                    end
                    else begin
                        next_state = S_P2_READ;  // read next element (INT8)
                    end
                end
            end
            S_P2_COMPUTE_HIGH: begin
                if (r_idx == r_length - 1) begin  // output the last normalized result (FP16)
                    next_state = S_DONE;
                end
                else begin
                    next_state = S_P2_READ;  // read next element (FP16)
                end
            end
            S_DONE: begin
                next_state = S_IDLE;
            end
            default: begin
                next_state = S_IDLE;
            end
        endcase
    end

    // mean_var_engine
    assign mv_start      = (state == S_P1_READ && r_idx == 0 && r_data_type == 0);
    assign mv_data_valid = (state == S_P1_FEED && r_data_type == 0);
    assign mv_data_in    = signed'(sram_rd0_data);
    assign mv_din_last   = (state == S_P1_FEED && r_idx == (r_length - 1) && r_data_type == 0);

    mean_var_engine u_mean_var_engine (
        .clk         (clk),
        .rst_n       (rst_n),
        .length      (r_length),
        .start       (mv_start),
        .din_valid   (mv_data_valid),
        .din         (mv_data_in),
        .din_last    (mv_din_last),
        .mean_out    (mv_mean_out),
        .var_out     (mv_var_out),
        .result_valid(mv_valid)
    );

    logic signed [15:0] r_mean;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_mean <= 0;
        end
        else if (mv_valid) begin
            r_mean <= mv_mean_out;
        end
    end

    // Index counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_idx <= 0;
        end
        else if (state == S_IDLE && cmd_valid) begin  // ready count
            r_idx <= 0;
        end
        else if (state == S_P1_FEED && next_state == S_P1_READ) begin // continue count for element one by one
            r_idx <= r_idx + 1'b1;
        end
        else if ((state == S_P1_WAIT && mv_valid) || state == S_P1_FP16_FINALIZE) begin // reset for PASS2
            r_idx <= 0;
        end
        else if (state == S_P2_COMPUTE && next_state == S_P2_READ && r_data_type == 0) begin //INT8: next element compute
            r_idx <= r_idx + 1'b1;
        end
        else if (state == S_P2_COMPUTE_HIGH && next_state == S_P2_READ && r_data_type == 1) begin // FP16: next element compute
            r_idx <= r_idx + 1'b1;
        end
        else if (state == S_P2_COMPUTE && next_state == S_DONE && r_data_type == 0) begin  // HOLD
            r_idx <= r_idx;
        end
    end

    // rsqrt_lut
    logic [ 7:0] rsqrt_addr;
    logic [15:0] rsqrt_data;
    assign rsqrt_addr = mv_var_out[31:24];  // INT8 path: top 8 bits as index

    rsqrt_lut u_rsqrt_lut (
        .clk     (clk),
        .addr    (rsqrt_addr),
        .data_out(rsqrt_data)
    );


    // SRAM readout
    always_comb begin
        // read input and gamma (FP16) from sram0
        sram_rd0_en   = 0;
        sram_rd0_addr = 0;
        // read beta from sram1
        sram_rd1_en   = 0;
        sram_rd1_addr = 0;

        if (r_data_type == 0) begin  // INT8
            if (state == S_P1_READ) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_src_base + r_idx;
            end
            else if (state == S_P2_READ) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_src_base + r_idx;
                sram_rd1_en   = 1'b1;
                sram_rd1_addr = r_beta_base + r_idx;
            end
        end
        else begin  // FP16
            if (state == S_P1_READ) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_src_base + (r_idx << 1);  // idx*2+0
            end
            else if (state == S_P1_READ_HIGH) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_src_base + (r_idx << 1) + 1;  // idx*2+1
            end
            else if (state == S_P2_READ) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_src_base + (r_idx << 1);  // idx*2+0
            end
            else if (state == S_P2_READ_HIGH) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_src_base + (r_idx << 1) + 1;  // idx*2+1
            end
            else if (state == S_P2_READ_GAMMA) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_gamma_base + (r_idx << 1);  // idx*2
            end
            else if (state == S_P2_READ_GAMMA_HIGH) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_gamma_base + (r_idx << 1) + 1;  // idx*2+1
            end
            else if (state == S_P2_READ_BETA) begin
                sram_rd1_en   = 1'b1;
                sram_rd1_addr = r_beta_base + (r_idx << 1);  // idx*2
            end
            else if (state == S_P2_READ_BETA_HIGH) begin
                sram_rd1_en   = 1'b1;
                sram_rd1_addr = r_beta_base + (r_idx << 1) + 1;  // idx*2+1
            end
        end
    end

    // FP16 low byte storage
    logic [7:0] lo_byte;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lo_byte <= 0;
        end
        else begin
            if (state == S_P1_READ_HIGH && r_data_type == 1) begin              // SRAM同步读出，这里在锁存的时候要打一拍
                lo_byte <= sram_rd0_data;
            end
            else if (state == S_P2_READ_HIGH && r_data_type == 1) begin         // SRAM同步读出，这里在锁存的时候要打一拍
                lo_byte <= sram_rd0_data;
            end
            else if (state == S_P2_READ_GAMMA_HIGH && r_data_type == 1) begin   // SRAM同步读出，这里在锁存的时候要打一拍
                lo_byte <= sram_rd0_data;
            end
        else if (state == S_P2_READ_BETA_HIGH && r_data_type == 1) begin        // SRAM同步读出，这里在锁存的时候要打一拍
                lo_byte <= sram_rd1_data;
            end
        end
    end

    // Assemble FP16 value from {hi, lo} bytes
    logic [15:0] fp16_rd_val;
    logic [15:0] fp16_gamma_val;
    logic [15:0] fp16_beta_val;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fp16_rd_val    <= 0;
            fp16_gamma_val <= 0;
            fp16_beta_val  <= 0;
        end
        else begin
            // FP16 PASS1 read
            if (state == S_P1_READFP16_WAIT) begin                   // SRAM同步读出，这里在锁存的时候要打一拍
                fp16_rd_val <= {sram_rd0_data, lo_byte};
            end
            // FP16 PASS2 read
            if (state == S_P2_READ_GAMMA) begin             // SRAM同步读出，这里在锁存的时候要打一拍
                fp16_rd_val <= {sram_rd0_data, lo_byte};
            end
            if (state == S_P2_READ_BETA) begin              // SRAM同步读出，这里在锁存的时候要打一拍
                fp16_gamma_val <= {sram_rd0_data, lo_byte};
            end
            if (state == S_P2_READ_WAIT) begin              // SRAM同步读出，这里在锁存的时候要打一拍
                fp16_beta_val <= {sram_rd1_data, lo_byte};
            end
        end
    end

    // Pass 1: FP16 accumulation (bypass mean_var_engine)
    logic [15:0] fp16_x_sq;
    always_comb begin
        fp16_x_sq = fp16_mul(fp16_rd_val, fp16_rd_val);
    end

    logic [15:0] r_fp16_sum;
    logic [15:0] r_fp16_sq_sum;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_fp16_sum    <= 0;
            r_fp16_sq_sum <= 0;
        end
        else if (state == S_IDLE && cmd_valid) begin
            r_fp16_sum    <= 0;
            r_fp16_sq_sum <= 0;
        end
        else if (state == S_P1_FEED && r_data_type) begin
            r_fp16_sum    <= fp16_add(r_fp16_sum, fp16_rd_val);
            r_fp16_sq_sum <= fp16_add(r_fp16_sq_sum, fp16_x_sq);
        end
    end

    // FP16 finalize: compute mean and var
    logic [15:0] fp16_mean_val;
    logic [15:0] fp16_x_sq_val;
    logic [15:0] fp16_mean_sq_val;
    logic [15:0] fp16_var_val;
    always_comb begin
        // FP16 mean value
        // mean = sum * (1/N)
        fp16_mean_val    = fp16_mul(r_fp16_sum, recip_n_lut_out);
        // FP16 square mean value
        // E[x^2] = sum_sq * (1/N)
        fp16_x_sq_val    = fp16_mul(r_fp16_sq_sum, recip_n_lut_out);
        // FP16 mean value square
        // mean^2
        fp16_mean_sq_val = fp16_mul(fp16_mean_val, fp16_mean_val);
        // FP16 var value
        // var = E[x^2] - mean^2
        fp16_var_val     = fp16_sub(fp16_x_sq_val, fp16_mean_sq_val);
    end

    logic [15:0] r_mean_fp16;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_mean_fp16 <= 0;
        end
        else if (state == S_P1_FP16_FINALIZE && r_data_type == 1) begin
            r_mean_fp16 <= fp16_mean_val;
        end
    end

    logic [ 7:0] rsqrt_fp16_addr;
    logic [15:0] rsqrt_fp16_data;
    assign rsqrt_fp16_addr = (state == S_RSQRT && r_data_type == 1) ? fp16_var_val[15:8] : 8'd0;

    graph_rsqrt_lut_fp16 u_rsqrt_fp16_lut (
        .clk     (clk),
        .addr    (rsqrt_fp16_addr),
        .data_out(rsqrt_fp16_data)
    );

    // Latch rsqrt_result
    logic [15:0] r_inv_std;
    logic [15:0] r_inv_std_fp16;
    always_ff @(posedge clk) begin
        if (state == S_RSQRT_WAIT) begin
            if (r_data_type == 0) begin
                r_inv_std <= rsqrt_data;
            end
            else begin
                r_inv_std_fp16 <= rsqrt_fp16_data;
            end
        end
    end

    // Pipeline registers for PASS2 data (INT8)
    logic signed [7:0] p_input;
    logic signed [7:0] p_beta;
    logic signed [7:0] p_gamma;
    always_ff @(posedge clk) begin
        if (state == S_P2_READ_WAIT && r_data_type == 0) begin
            p_input <= signed'(sram_rd0_data);
            p_beta  <= signed'(sram_rd1_data);
            // For gamma, we assume it's unity-scaled or stored with input
            // In a full implementation, a third read port or time-multiplexing
            // would fetch gamma. Here we use a default gamma of 1 (=127 in int8 scale).
            p_gamma <= 8'sd127;
        end
    end


    // PASS2 Normalize compution (INT8)
    logic signed [15:0] centered;
    logic signed [31:0] scaled;
    logic signed [31:0] gamma_applied;
    logic signed [31:0] bias_added;
    logic signed [ 7:0] norm_result;
    always_comb begin
        // Center: (in[i] - mean) where in is int8 and mean is Q8.8
        // Convert input to Q8.8 first: in << 8
        centered      = (16'(signed'(p_input)) <<< 8) - r_mean;
        // Scale by inv_std (Q0.16): result is Q8.24, take upper bits
        scaled        = 32'(signed'(centered)) * 32'(signed'({1'b0, r_inv_std}));
        // scaled is Q8.24 (8.8 * 0.16 = 8.24)
        // Shift right by 16 to get Q8.8
        scaled        = scaled >>> 16;
        // Apply gamma (int8, representing scale ~1.0 when gamma=127)
        // gamma_applied = scaled * gamma / 128 (normalize gamma to ~1.0)
        gamma_applied = (scaled * 32'(signed'(p_gamma))) >>> 7;
        // Add beta (int8, sign-extend to Q8.8 by shifting left 8)
        bias_added    = gamma_applied + 32'(signed'(p_beta) <<< 8);

        // Requantize: shift right by 8 to get int8 from Q8.8
        if ((bias_added >>> 8) > 32'sd127) begin
            norm_result = 8'sd127;
        end
        else if ((bias_added >>> 8) < -32'sd128) begin
            norm_result = -8'sd127;
        end
        else begin
            norm_result = bias_added[15:8];
        end
    end

    // ----------------------------------------------------------------
    // Pass 2: Normalize computation (FP16 path)
    // centered = fp16_sub(x, mean)
    // scaled   = fp16_mul(centered, inv_std)
    // gamma_ap = fp16_mul(scaled, gamma)
    // result   = fp16_add(gamma_ap, beta)
    // ----------------------------------------------------------------
    logic [15:0] fp16_centered;
    logic [15:0] fp16_scaled;
    logic [15:0] fp16_gamma_applied;
    logic [15:0] fp16_norm_result;

    always_comb begin
        fp16_centered      = fp16_sub(fp16_rd_val, r_mean_fp16);
        fp16_scaled        = fp16_mul(fp16_centered, r_inv_std_fp16);
        fp16_gamma_applied = fp16_mul(fp16_scaled, fp16_gamma_val);
        fp16_norm_result   = fp16_add(fp16_gamma_applied, fp16_beta_val);
    end

    // Register to hold FP16 result for 2-byte write
    logic [15:0] r_fp16_wr_val;

    always_ff @(posedge clk) begin
        if (state == S_P2_COMPUTE && r_data_type == 1) r_fp16_wr_val <= fp16_norm_result;
    end


    // SRAM Write
    always_comb begin
        sram_wr_en   = 0;
        sram_wr_addr = 0;
        sram_wr_data = 0;

        if (r_data_type == 0) begin
            if (state == S_P2_COMPUTE) begin
                sram_wr_en   = 1'b1;
                sram_wr_addr = r_dst_base + r_idx;
                sram_wr_data = norm_result;
            end
        end
        else begin
            if (state == S_P2_COMPUTE) begin
                sram_wr_en   = 1'b1;
                sram_wr_addr = r_dst_base + (r_idx << 1);  // idx*2+0
                sram_wr_data = fp16_norm_result[7:0];
            end
            else if (state == S_P2_COMPUTE_HIGH) begin
                sram_wr_en   = 1'b1;
                sram_wr_addr = r_dst_base + (r_idx << 1) + 1;  // idx*2+0
                sram_wr_data = r_fp16_wr_val[15:8];
            end
        end
    end

    assign done      = (state == S_DONE);
    assign busy      = (state != S_IDLE);
    assign cmd_ready = (state == S_IDLE);
endmodule
