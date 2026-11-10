// Full grand piano: strings, soundboard, sympathetic strings and room.
`default_nettype none

module grand_piano_top
  import grand_piano_pkg::*;
#(
  parameter int SAMPLE_PERIOD = 2083
) (
  input  wire                clk,
  input  wire                rst,

  input  wire                note_on_valid,
  input  wire [6:0]          note_on_pitch,
  input  wire [6:0]          note_on_velocity,
  input  wire                note_off_valid,
  input  wire [6:0]          note_off_pitch,
  input  wire                sustain_pedal,

  output wire                note_on_ready,
  output wire                event_overflow,

  output logic               sample_valid,
  output logic signed [23:0] sample_l,
  output logic signed [23:0] sample_r
);

  logic                      dry_valid;
  logic signed [STATE_W-1:0] dry_l, dry_r;
  logic [127:0]              pitch_active;

  grand_piano_core #(
    .SAMPLE_PERIOD (SAMPLE_PERIOD)
  ) u_strings (
    .clk              (clk),
    .rst              (rst),
    .note_on_valid    (note_on_valid),
    .note_on_pitch    (note_on_pitch),
    .note_on_velocity (note_on_velocity),
    .note_off_valid   (note_off_valid),
    .note_off_pitch   (note_off_pitch),
    .sustain_pedal    (sustain_pedal),
    .note_on_ready    (note_on_ready),
    .event_overflow   (event_overflow),
    .sample_valid     (dry_valid),
    .dry_l            (dry_l),
    .dry_r            (dry_r),
    .pitch_active     (pitch_active)
  );

  grand_piano_fx u_fx (
    .clk           (clk),
    .rst           (rst),
    .in_valid      (dry_valid),
    .dry_l         (dry_l),
    .dry_r         (dry_r),
    .sustain_pedal (sustain_pedal),
    .pitch_active  (pitch_active),
    .out_valid     (sample_valid),
    .out_l         (sample_l),
    .out_r         (sample_r)
  );

endmodule

`default_nettype wire
