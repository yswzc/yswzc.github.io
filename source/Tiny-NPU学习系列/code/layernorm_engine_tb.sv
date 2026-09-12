// `include "../rtl/ops/vec_engine.sv"
// `include "../rtl/mem/sram_dp.sv"
module layernorm_engine_tb;

    parameter CLK_PERIOD = 10;
    parameter DATA_WIDTH = 8;
    parameter ADDR_WIDTH = 16;
    integer                       incorrect_num;

    //Ports
    logic                         clk;
    logic                         rst_n;
    logic                         cmd_valid;
    logic                         cmd_ready;
    logic                         data_type;
    logic        [          15:0] length;
    logic        [          15:0] src_base;
    logic        [          15:0] dst_base;
    logic        [          15:0] gamma_base;
    logic        [          15:0] beta_base;
    logic                         sram_rd0_en;
    logic        [          15:0] sram_rd0_addr;
    logic signed [DATA_WIDTH-1:0] sram_rd0_data;
    logic                         sram_rd1_en;
    logic        [          15:0] sram_rd1_addr;
    logic signed [DATA_WIDTH-1:0] sram_rd1_data;
    logic                         sram_wr_en;
    logic        [          15:0] sram_wr_addr;
    logic signed [DATA_WIDTH-1:0] sram_wr_data;
    logic                         busy;
    logic                         done;

    // logic [DATA_WIDTH-1:0] sram0_mem [0:2**ADDR_WIDTH-1];
    // logic [DATA_WIDTH-1:0] sram1_mem [0:2**ADDR_WIDTH-1];
    // logic [DATA_WIDTH-1:0] sram_mem [0:2**ADDR_WIDTH-1];

    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    // 模拟SRAM
    // always_ff @(posedge clk) begin
    //     if (sram_rd0_en) begin
    //         sram_rd0_data <= sram0_mem[sram_rd0_addr];
    //     end
    //     if (sram_rd1_en) begin
    //         sram_rd1_data <= sram1_mem[sram_rd1_addr];
    //     end
    //     if (sram_wr_en) begin
    //         sram_mem[sram_wr_addr] <= sram_wr_data;
    //     end
    // end

    // 真实SRAM
    logic sram_wr0_en, sram_wr1_en, sram_we0_en, sram_we1_en;
    logic signed [DATA_WIDTH-1:0] sram_wr0_data, sram_wr1_data;
    logic [ADDR_WIDTH-1:0] sram_wr0_addr, sram_wr1_addr;
    sram_dp #(
        .WIDTH (DATA_WIDTH),
        .ADDR_W(ADDR_WIDTH),
        .DEPTH (2 ** ADDR_WIDTH)
    ) sram_rd0_inst (
        .clk   (clk),
        .en_a  (sram_rd0_en),
        .we_a  ('0),
        .addr_a(sram_rd0_addr),
        .din_a ('0),
        .dout_a(sram_rd0_data),

        .en_b  (sram_wr0_en),
        .we_b  (sram_we0_en),
        .addr_b(sram_wr0_addr),
        .din_b (sram_wr0_data),
        .dout_b()
    );

    sram_dp #(
        .WIDTH (DATA_WIDTH),
        .ADDR_W(ADDR_WIDTH),
        .DEPTH (2 ** ADDR_WIDTH)
    ) sram_rd1_inst (
        .clk   (clk),
        .en_a  (sram_rd1_en),
        .we_a  ('0),
        .addr_a(sram_rd1_addr),
        .din_a ('0),
        .dout_a(sram_rd1_data),

        .en_b  (sram_wr1_en),
        .we_b  (sram_we1_en),
        .addr_b(sram_wr1_addr),
        .din_b (sram_wr1_data),
        .dout_b()
    );

    sram_dp #(
        .WIDTH (DATA_WIDTH),
        .ADDR_W(ADDR_WIDTH),
        .DEPTH (2 ** ADDR_WIDTH)
    ) sram_wr_inst (
        .clk   (clk),
        .en_a  (),
        .we_a  (),
        .addr_a(),
        .din_a (),
        .dout_a(),

        .en_b  (sram_wr_en),
        .we_b  (sram_wr_en),
        .addr_b(sram_wr_addr),
        .din_b (sram_wr_data),
        .dout_b()
    );


    layernorm_engine layernorm_engine_inst (
        .clk          (clk),
        .rst_n        (rst_n),
        .cmd_valid    (cmd_valid),
        .cmd_ready    (cmd_ready),
        .data_type    (data_type),
        .length       (length),
        .src_base     (src_base),
        .dst_base     (dst_base),
        .gamma_base   (gamma_base),
        .beta_base    (beta_base),
        .sram_rd0_en  (sram_rd0_en),
        .sram_rd0_addr(sram_rd0_addr),
        .sram_rd0_data(sram_rd0_data),
        .sram_rd1_en  (sram_rd1_en),
        .sram_rd1_addr(sram_rd1_addr),
        .sram_rd1_data(sram_rd1_data),
        .sram_wr_en   (sram_wr_en),
        .sram_wr_addr (sram_wr_addr),
        .sram_wr_data (sram_wr_data),
        .busy         (busy),
        .done         (done)
    );

    task automatic reset();
        begin
            rst_n      = 0;
            cmd_valid  = 0;
            data_type  = 0;
            length     = 0;
            src_base   = 0;
            dst_base   = 0;
            gamma_base = 0;
            beta_base  = 0;
            @(posedge clk);
            rst_n = 1;
        end
    endtask

    //   task automatic sram0_write(
    //     input logic [ADDR_WIDTH-1:0] addr,
    //     input logic [DATA_WIDTH-1:0] data
    //   );
    //     begin
    //         sram0_mem[addr] = data;
    //     end
    //   endtask

    //   task automatic sram1_write(
    //     input logic [ADDR_WIDTH-1:0] addr,
    //     input logic [DATA_WIDTH-1:0] data
    //   );
    //     begin
    //         sram1_mem[addr] = data;
    //     end
    //   endtask

    task automatic actual_sram0_write(input logic [ADDR_WIDTH-1:0] addr,
                                      input logic [DATA_WIDTH-1:0] data);
        begin
            sram_wr0_addr = addr;
            sram_wr0_data = data;
        end
    endtask

    task automatic actual_sram1_write(input logic [ADDR_WIDTH-1:0] addr,
                                      input logic [DATA_WIDTH-1:0] data);
        begin
            sram_wr1_addr = addr;
            sram_wr1_data = data;
        end
    endtask

    task automatic initial_sram_rd(input integer len);
        begin
            integer i;
            sram_wr0_en = 1;
            sram_wr1_en = 1;
            sram_we0_en = 1;
            sram_we1_en = 1;
            for (i = 0; i < len; i = i + 1) begin
                actual_sram0_write(src_base + i, i);        // write src
                actual_sram1_write(src_base + i, 8'b0);        // write beta
                @(posedge clk);
            end
            for (i = len; i < 2*len; i = i + 1) begin
                actual_sram0_write(src_base + i, 8'b1);        // write gamma
                @(posedge clk);
            end
            sram_wr0_en = 0;
            sram_wr1_en = 0;
            sram_we0_en = 0;
            sram_we1_en = 0;
        end
    endtask

    task automatic operation(input logic dtype, input logic [ADDR_WIDTH-1:0] len,
                             input logic [ADDR_WIDTH-1:0] src, input logic [ADDR_WIDTH-1:0] dst,
                             input logic [ADDR_WIDTH-1:0] gamma, input logic [ADDR_WIDTH-1:0] beta);
        begin
            @(posedge clk);
            cmd_valid  = 1;
            data_type  = dtype;
            length     = len;
            src_base   = src;
            dst_base   = dst;
            gamma_base = gamma;
            beta_base  = beta;
            @(posedge clk);
            cmd_valid = 0;
            wait (done);
        end
    endtask

    // task automatic expect_result(
    //     input logic [1:0] op, input logic [DATA_WIDTH-1:0] src0_data,
    //     input logic [DATA_WIDTH-1:0] src1_data, input logic [DATA_WIDTH-1:0] scl,
    //     input logic [DATA_WIDTH-1:0] shift_value, output logic [DATA_WIDTH-1:0] result);
    //     begin
    //         logic signed [2*DATA_WIDTH-1:0] add_result, mul_result, scale_shifted, mul_round;
    //         logic signed [4*DATA_WIDTH-1:0] scale_result;

    //         add_result    = 16'(signed'(src0_data)) + 16'(signed'(src1_data));
    //         mul_result    = 16'(signed'(src0_data)) * 16'(signed'(src1_data));
    //         scale_result  = 32'(signed'(src0_data)) * 32'(signed'({1'b0, scl}));
    //         scale_shifted = 16'(scale_result >>> shift_value);
    //         mul_round     = (mul_result + 16'sd64) >>> 7;

    //         case (op)
    //             2'b00: begin
    //                 if (add_result > 16'sd127) result = 8'sd127;
    //                 else if (add_result < -16'sd128) result = -8'sd128;
    //                 else result = add_result[7:0];
    //             end
    //             2'b01: begin
    //                 if (mul_round > 16'sd127) result = 8'sd127;
    //                 else if (mul_round < -16'sd128) result = -8'sd128;
    //                 else result = mul_round[7:0];
    //             end
    //             2'b10: begin
    //                 if (scale_shifted > 16'sd127) result = 8'sd127;
    //                 else if (scale_shifted < -16'sd128) result = -8'sd128;
    //                 else result = scale_shifted[7:0];
    //             end
    //             2'b11: begin
    //                 result = src0_data;
    //             end
    //             default: begin
    //                 result = '0;
    //             end
    //         endcase
    //     end
    // endtask

    // task automatic check_result(input logic [ADDR_WIDTH-1:0] dst, input logic [ADDR_WIDTH-1:0] len,
    //                             input logic [DATA_WIDTH-1:0] expected_data[0:2**ADDR_WIDTH-1]);
    //     integer i;

    //     begin
    //         for (i = 0; i < len; i = i + 1) begin
    //             if (sram_wr_inst.mem[dst+i] !== expected_data[i]) begin
    //                 incorrect_num = incorrect_num + 1;
    //                 $display("Mismatch at address %0d: expected %0d, got %0d", dst + i,
    //                          signed'(expected_data[i]), signed'(sram_wr_inst.mem[dst+i]));
    //             end
    //             else begin
    //                 $display("Match at address %0d: value %0d", dst + i,
    //                          signed'(sram_wr_inst.mem[dst+i]));
    //             end
    //         end
    //         if (incorrect_num != 0) begin
    //             $display("Mismatch number is: %0d", incorrect_num);
    //         end
    //         else begin
    //             $display("Test PASSED !!!");
    //         end
    //     end
    // endtask

    integer                  i;
    integer                  len_value_tb;
    logic                    date_type_tb;
    logic   [DATA_WIDTH-1:0] expect_result_array[0:2**ADDR_WIDTH-1];
    initial begin
        reset();
        incorrect_num = 0;
        len_value_tb  = 100;
        repeat (10) @(posedge clk);
        // for (i=0; i<len_value_tb; i=i+1) begin
        //     sram0_write(src0_base + i, ($random%256)-128);
        //     sram1_write(src1_base + i, ($random%256)-128);
        //     @(posedge clk);
        // end
        initial_sram_rd(len_value_tb);
        repeat (10) @(posedge clk);
        $display("SRAM initialized with random data. DONE!!!!");
        date_type_tb = 1'b0;  // INT8  
        fork
            begin
                operation(date_type_tb, len_value_tb, 16'd0, 16'd0, 16'd100, 16'd0);  // INT8 Layer Normalization
            end
            begin
                while (!done) begin
                    @(posedge clk)
                    if (sram_wr_en) begin
                        $display("addr[%0d]: %0d", sram_wr_addr, signed'(sram_wr_data));
                    end
                end
            end
        join
        repeat (10) @(posedge clk);
        $display("INT8 Layer Normalization Test. DONE!!!!");
        $finish;
    end


endmodule
;
