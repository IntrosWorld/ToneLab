// Shared types and helpers for the grand piano engine.
package grand_piano_pkg;

  `include "grand_piano_params.svh"

  // Reverb delay lengths, mutually prime so repeats never line up into a pitch.
  localparam int COMB_LEN [8] = '{1214, 1293, 1390, 1476, 1548, 1623, 1695, 1760};
  localparam int ALLPASS_LEN [4] = '{605, 480, 371, 245};

  // Saturating narrow: keeps a runaway resonator from wrapping around into a
  // full-scale click, which is the loudest possible failure in an audio path.
  function automatic logic signed [STATE_W-1:0] sat_state(input logic signed [63:0] v);
    localparam logic signed [63:0] HI =  (64'sd1 <<< (STATE_W-1)) - 64'sd1;
    localparam logic signed [63:0] LO = -(64'sd1 <<< (STATE_W-1));
    if (v > HI)      sat_state = HI[STATE_W-1:0];
    else if (v < LO) sat_state = LO[STATE_W-1:0];
    else             sat_state = v[STATE_W-1:0];
  endfunction

  function automatic logic signed [23:0] sat_pcm(input logic signed [63:0] v);
    localparam logic signed [63:0] HI =  64'sd8388607;
    localparam logic signed [63:0] LO = -64'sd8388608;
    if (v > HI)      sat_pcm = HI[23:0];
    else if (v < LO) sat_pcm = LO[23:0];
    else             sat_pcm = v[23:0];
  endfunction

endpackage
