// mean_var_engine.sv
// ----------------------------------------------------------------
// Implementation of Mean and Variance Engine
// Implementation of Bitwise Division
module mean_var_engine (
    input logic clk,
    input logic rst_n,

    input logic        start,  // begin to accumulation
    input logic [15:0] length, // number of elements in the vector

    input logic signed [7:0] din,  // input 8-bit signed data 
    input logic din_valid,  // input data valid
    input logic din_last,  // input data last (indicates the last element of the vector)

    output logic signed [15:0] mean_out,     // Q8.8 fixed-point mean result
    output logic        [31:0] var_out,      // Q16.16 unsigned fixed-point variance result
    output logic               result_valid  // output result valid
);

    // divide signal
    logic signed [47:0] div_remainder;
    logic signed [47:0] div_dividend;
    logic signed [47:0] div_quotient;
    logic div_negative; // 1: negative 0: positive
    logic [5:0] div_count;
    logic [47:0] sq_div_dividend;
    logic [47:0] sq_div_remainder;
    logic [47:0] sq_div_quotient;
    logic div_phase; // 0: sum division 1: sq_sum division

    typedef enum logic [2:0] {
        S_IDLE,
        S_ACCUM,
        S_DIVIDE,
        S_VARIANCE,
        S_DONE
    } state_t;
    state_t state, state_next;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else begin
            state <= state_next;
        end
    end

    always_comb begin
        state_next = state;
        case (state)
            S_IDLE: if (start) state_next = S_ACCUM;
            S_ACCUM: if(din_valid && din_last) state_next = S_DIVIDE;
            S_DIVIDE: if (div_count == 0 && div_phase) state_next = S_VARIANCE;
            S_VARIANCE: state_next = S_DONE;
            S_DONE: state_next = S_IDLE;
            default: state_next = S_IDLE;
        endcase
    end

    // extend input and compute input square
    logic signed [15:0] din_ext;
    logic [15:0] din_sq;

    always_comb begin
        din_ext = 16'(signed'(din));
        din_sq = 16'(din) * 16'(din);
    end

    logic [15:0] r_length;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_length <= '0;
        end else if (start) begin
            r_length <= length;
        end
    end

    // accumulation signals
    logic signed [31:0] r_sum;
    logic [47:0] r_sum_sq;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_sum <= '0;
            r_sum_sq <= '0;
        end else if (state == S_IDLE) begin
            r_sum <= '0;
            r_sum_sq <= '0;
        end else if (state == S_ACCUM && din_valid) begin
            r_sum <= r_sum + 32'(din_ext);
            r_sum_sq <= r_sum_sq + 48'(din_sq);
        end
    end

    // Division process
    logic signed [15:0] r_mean;
    logic [31:0] r_var;

    logic signed [47:0] remainder_sub;
    logic signed [47:0] sq_remainder_sub;
    logic signed [47:0] r_div_dividend;
    logic [47:0] r_div_dividend_abs;
    logic div_dividend_negative;
    logic signed [47:0] r_sq_div_dividend;

    always_comb begin
        remainder_sub = div_remainder - 48'(r_length);
        sq_remainder_sub = sq_div_remainder - 48'(r_length);
        // Prepare dividend for mean and variance division
        // parpare the abs of div_dividend for mean division
        r_div_dividend = 48'(signed'(r_sum + 32'(din_ext))) << 8;
        div_dividend_negative = r_div_dividend[47];
        if (div_dividend_negative) begin
            r_div_dividend_abs = (~$unsigned(r_div_dividend)) + 1;
        end else begin
            r_div_dividend_abs = $unsigned(r_div_dividend);
        end
        // prepare the dividend for variance division
        r_sq_div_dividend =  48'(r_sum_sq + 48'(din_sq)) << 16;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_mean <= '0;
            div_dividend <= '0;
            div_remainder <= '0;
            div_quotient <= '0;
            div_count <= '0;
            div_negative <= '0;
            div_phase <= 0;
            sq_div_dividend <= '0;
            sq_div_remainder <= '0;
            sq_div_quotient <= '0;
        end else if (state == S_ACCUM && din_valid && din_last) begin
            div_dividend <= r_div_dividend_abs;
            div_remainder <= {47'(0),r_div_dividend_abs[47]};
            div_count <= 6'd48;
            div_quotient <= '0;
            div_phase <= 0;
            div_negative <= div_dividend_negative;

            sq_div_dividend <= r_sq_div_dividend;
            sq_div_remainder <= {47'(0),r_sq_div_dividend[47]};
            sq_div_quotient <= '0;
        end else if (state == S_DIVIDE) begin
            if (!div_phase) begin
                if (div_count > 0) begin
                    if(div_remainder >= 48'(r_length)) begin
                        div_quotient <= {div_quotient[46:0], 1'b1};
                        if (div_count > 1) begin
                            div_remainder <= {remainder_sub[46:0], div_dividend[div_count-2]};
                        end
                    end else begin
                        div_quotient <= {div_quotient[46:0], 1'b0};
                         if (div_count > 1) begin
                            div_remainder <= {div_remainder[46:0], div_dividend[div_count-2]};
                         end
                    end
                    div_count <= div_count - 1;
                end else begin
                    if (div_negative) begin
                        r_mean <= -$signed(div_quotient[15:0]); // Q8.8 fixed-point mean result
                    end else begin
                        r_mean <= $signed(div_quotient[15:0]); // Q8.8 fixed-point mean result
                    end
                    div_phase <= 1;
                    div_count <= 6'd48;
                end
            end else begin
                if (div_count > 0) begin
                    if(sq_div_remainder >= 48'(r_length)) begin
                        sq_div_quotient <= {sq_div_quotient[46:0], 1'b1};
                        if (div_count > 1) begin
                            sq_div_remainder <= {sq_remainder_sub[46:0],  sq_div_dividend[div_count-2]};
                        end
                    end else begin
                        sq_div_quotient <= {sq_div_quotient[46:0], 1'b0};
                         if (div_count > 1) begin
                            sq_div_remainder <= {sq_div_remainder[46:0], sq_div_dividend[div_count-2]};
                         end
                    end
                    div_count <= div_count - 1;
                end
            end
        end
    end


    // Variance computation: var = E[x^2] - (mean)^2
    logic [31:0] mean_sq;

    always_comb begin
        mean_sq = 32'(r_mean) * 32'(r_mean); 
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_var <= '0;
        end else if (state == S_VARIANCE) begin
            if (sq_div_quotient[31:0] >= mean_sq[31:0]) begin
                r_var <= sq_div_quotient[31:0] - mean_sq[31:0]; // Q16.16 unsigned fixed-point variance result
            end else begin
                r_var <= '0;  // numerical guard against negative variance
            end
        end
    end

    assign mean_out = r_mean;
    assign var_out = r_var;
    assign result_valid = (state == S_DONE);

endmodule
