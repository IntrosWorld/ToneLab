// Everything between the strings and the listener.
//
//   dry strings -> soundboard body modes -> sympathetic strings -> room
//
// All three stages are resonator banks or delay networks, and they run inside
// the same audio-sample period as the string engine.  Every resonator here
// uses the same magic-circle form as the strings, for the same reason: a body
// mode with Q = 70 at 46 Hz has r = 0.99996, which is exactly as degenerate in
// a direct form as a long string partial is.

`default_nettype none

module grand_piano_fx
  import grand_piano_pkg::*;
(
  input  wire                       clk,
  input  wire                       rst,
  input  wire                       in_valid,
  input  wire signed [STATE_W-1:0]  dry_l,
  input  wire signed [STATE_W-1:0]  dry_r,
  input  wire                       sustain_pedal,
  input  wire [127:0]               pitch_active,

  output logic                      out_valid,
  output logic signed [23:0]        out_l,
  output logic signed [23:0]        out_r
);

  // =====================================================================
  // Soundboard body modes
  // =====================================================================
  logic signed [COEF_W-1:0]  body_k    [BODY_MODES];
  logic signed [LOSS_W-1:0]  body_loss [BODY_MODES];
  logic signed [COEF_W-1:0]  body_gl   [BODY_MODES];
  logic signed [COEF_W-1:0]  body_gr   [BODY_MODES];

  logic signed [STATE_W-1:0] body_xl [BODY_MODES], body_yl [BODY_MODES];
  logic signed [STATE_W-1:0] body_xr [BODY_MODES], body_yr [BODY_MODES];

  // =====================================================================
  // Sympathetic strings
  // =====================================================================
  logic signed [COEF_W-1:0]  symp_k    [SYMP_SLOTS];
  logic signed [LOSS_W-1:0]  symp_loss [SYMP_SLOTS];
  logic signed [COEF_W-1:0]  symp_gain [SYMP_SLOTS];
  logic signed [GAIN_W-1:0]  symp_pl   [SYMP_SLOTS];
  logic signed [GAIN_W-1:0]  symp_pr   [SYMP_SLOTS];
  logic        [7:0]         symp_und  [SYMP_SLOTS];
  logic        [7:0]         symp_pitch[SYMP_SLOTS];

  logic signed [STATE_W-1:0] symp_x [SYMP_SLOTS], symp_y [SYMP_SLOTS];

  `include "grand_piano_fx_roms.svh"

  initial begin
    build_body_rom();
    build_symp_rom();
  end

  // =====================================================================
  // Room: parallel damped combs into series allpass diffusers
  // =====================================================================
  logic signed [STATE_W-1:0] comb_mem_l [NUM_COMBS][REVERB_LEN];
  logic signed [STATE_W-1:0] comb_mem_r [NUM_COMBS][REVERB_LEN];
  logic signed [STATE_W-1:0] comb_lp_l  [NUM_COMBS];
  logic signed [STATE_W-1:0] comb_lp_r  [NUM_COMBS];
  logic signed [STATE_W-1:0] ap_mem_l   [NUM_ALLPASS][REVERB_LEN];
  logic signed [STATE_W-1:0] ap_mem_r   [NUM_ALLPASS][REVERB_LEN];
  logic [$clog2(REVERB_LEN)-1:0] rev_ptr;

  // High-pass state for the sympathetic drive.
  logic signed [STATE_W-1:0] hp_prev, hp_state;

  function automatic logic signed [STATE_W-1:0] scale_g(
      input logic signed [STATE_W-1:0] v, input logic signed [GAIN_W-1:0] g);
    scale_g = sat_state((64'($signed(v)) * 64'($signed(g))) >>> GAIN_Q);
  endfunction

  always_ff @(posedge clk) begin
    if (rst) begin
      out_valid <= 1'b0;
      out_l     <= '0;
      out_r     <= '0;
      rev_ptr   <= '0;
      hp_prev   <= '0;
      hp_state  <= '0;
      for (int i = 0; i < BODY_MODES; i++) begin
        body_xl[i] <= '0; body_yl[i] <= '0;
        body_xr[i] <= '0; body_yr[i] <= '0;
      end
      for (int i = 0; i < SYMP_SLOTS; i++) begin
        symp_x[i] <= '0; symp_y[i] <= '0;
      end
      for (int c = 0; c < NUM_COMBS; c++) begin
        comb_lp_l[c] <= '0; comb_lp_r[c] <= '0;
        for (int d = 0; d < REVERB_LEN; d++) begin
          comb_mem_l[c][d] <= '0; comb_mem_r[c][d] <= '0;
        end
      end
      for (int a = 0; a < NUM_ALLPASS; a++)
        for (int d = 0; d < REVERB_LEN; d++) begin
          ap_mem_l[a][d] <= '0; ap_mem_r[a][d] <= '0;
        end
    end else begin
      out_valid <= 1'b0;

      if (in_valid) begin
        automatic logic signed [63:0] acc_l, acc_r, und_l, und_r;
        // Everything downstream runs at a quarter scale.  A reverb comb has a
        // DC gain of 1/(1-feedback) = 7.14, so with a dry peak of 2.83 its
        // stored state reaches 20.2 against the +-8.00 that Q3.28 holds -- the
        // comb memory saturates, and clipping inside a feedback loop makes
        // harmonics.  That was heard as midrange congestion masking the notes.
        // The master gain is raised by the same factor, so the output level is
        // unchanged and only the headroom differs.
        automatic logic signed [STATE_W-1:0] in_l, in_r;
        automatic logic signed [STATE_W-1:0] board_l, board_r;
        automatic logic signed [STATE_W-1:0] mixed_l, mixed_r;
        automatic logic signed [STATE_W-1:0] drive, hp;
        automatic logic signed [63:0] wet_l, wet_r;
        automatic logic signed [STATE_W-1:0] pre_l, pre_r;

        in_l = $signed(dry_l) >>> FX_HEADROOM_SHIFT;
        in_r = $signed(dry_r) >>> FX_HEADROOM_SHIFT;

        // -- soundboard -------------------------------------------------
        acc_l = '0;
        acc_r = '0;
        for (int i = 0; i < BODY_MODES; i++) begin
          automatic logic signed [STATE_W-1:0] xl, yl, xr, yr;
          xl = sat_state(64'($signed(body_xl[i]))
               + (($signed(body_k[i]) * $signed(body_yl[i])) >>> COEF_Q)
               + (($signed(body_gl[i]) * $signed(in_l)) >>> COEF_Q));
          yl = sat_state(64'($signed(body_yl[i]))
               - (($signed(body_k[i]) * $signed(xl)) >>> COEF_Q));
          xl = sat_state(64'($signed(xl)) - (($signed(body_loss[i]) * $signed(xl)) >>> LOSS_Q));
          yl = sat_state(64'($signed(yl)) - (($signed(body_loss[i]) * $signed(yl)) >>> LOSS_Q));
          body_xl[i] <= xl;
          body_yl[i] <= yl;
          acc_l = acc_l + 64'($signed(yl));

          xr = sat_state(64'($signed(body_xr[i]))
               + (($signed(body_k[i]) * $signed(body_yr[i])) >>> COEF_Q)
               + (($signed(body_gr[i]) * $signed(in_r)) >>> COEF_Q));
          yr = sat_state(64'($signed(body_yr[i]))
               - (($signed(body_k[i]) * $signed(xr)) >>> COEF_Q));
          xr = sat_state(64'($signed(xr)) - (($signed(body_loss[i]) * $signed(xr)) >>> LOSS_Q));
          yr = sat_state(64'($signed(yr)) - (($signed(body_loss[i]) * $signed(yr)) >>> LOSS_Q));
          body_xr[i] <= xr;
          body_yr[i] <= yr;
          acc_r = acc_r + 64'($signed(yr));
        end
        board_l = sat_state(acc_l);
        board_r = sat_state(acc_r);

        // Body mix: mostly straight off the strings, a fifth through the board.
        mixed_l = sat_state(64'($signed(in_l))
                  - (64'($signed(scale_g(in_l, GAIN_W'(BODY_MIX_G)))))
                  + 64'($signed(scale_g(board_l, GAIN_W'(BODY_MIX_G)))));
        mixed_r = sat_state(64'($signed(in_r))
                  - (64'($signed(scale_g(in_r, GAIN_W'(BODY_MIX_G)))))
                  + 64'($signed(scale_g(board_r, GAIN_W'(BODY_MIX_G)))));

        // -- sympathetic strings ----------------------------------------
        // Driven through a high-pass: coupling runs through the bridge, which
        // is a poor path for the lowest frequencies, and letting the bass
        // drive the bank just piles low energy back into the mix as mud.
        drive = sat_state((64'($signed(mixed_l)) + 64'($signed(mixed_r))) >>> 1);
        hp    = sat_state(64'($signed(drive)) - 64'($signed(hp_prev))
                + 64'($signed(scale_g(hp_state, GAIN_W'(SYMP_HP_A)))));
        hp_prev  <= drive;
        hp_state <= hp;

        acc_l = '0;
        acc_r = '0;
        und_l = '0;
        und_r = '0;
        for (int i = 0; i < SYMP_SLOTS; i++) begin
          automatic logic signed [STATE_W-1:0] sx, sy, u;
          // Pedal down drives everything; the top strings have no damper and
          // ring whether the pedal is down or not.  But a string that is
          // currently being played is not free to resonate -- it is already
          // sounding as a struck note.
          u  = (sustain_pedal || symp_und[i][0]) ? hp : '0;
          if (pitch_active[symp_pitch[i]]) u = '0;
          sx = sat_state(64'($signed(symp_x[i]))
               + (($signed(symp_k[i]) * $signed(symp_y[i])) >>> COEF_Q)
               + (($signed(symp_gain[i]) * $signed(u)) >>> COEF_Q));
          sy = sat_state(64'($signed(symp_y[i]))
               - (($signed(symp_k[i]) * $signed(sx)) >>> COEF_Q));
          sx = sat_state(64'($signed(sx)) - (($signed(symp_loss[i]) * $signed(sx)) >>> LOSS_Q));
          sy = sat_state(64'($signed(sy)) - (($signed(symp_loss[i]) * $signed(sy)) >>> LOSS_Q));
          // Pedal up: the dampers land on these strings and stop them.  Only
          // ceasing to drive them leaves them ringing for their full T60 --
          // 21.7 s at A#2 -- so the wash outlives the music.  The top strings
          // have no damper and are exempt.
          if (!sustain_pedal && !symp_und[i][0]) begin
            sx = scale_g(sx, GAIN_W'(SYMP_DAMP_G));
            sy = scale_g(sy, GAIN_W'(SYMP_DAMP_G));
          end
          symp_x[i] <= sx;
          symp_y[i] <= sy;
          // The always-on top strings are accumulated apart from the pedalled
          // bank so their level can be scaled independently.  At full level
          // they are an audible chime on unpedalled material.
          if (symp_und[i][0]) begin
            und_l = und_l + 64'($signed(scale_g(sy, symp_pl[i])));
            und_r = und_r + 64'($signed(scale_g(sy, symp_pr[i])));
          end else begin
            acc_l = acc_l + 64'($signed(scale_g(sy, symp_pl[i])));
            acc_r = acc_r + 64'($signed(scale_g(sy, symp_pr[i])));
          end
        end
        acc_l = acc_l + 64'($signed(scale_g(sat_state(und_l),
                                            GAIN_W'(UNDAMPED_ALWAYS_G))));
        acc_r = acc_r + 64'($signed(scale_g(sat_state(und_r),
                                            GAIN_W'(UNDAMPED_ALWAYS_G))));
        pre_l = sat_state(64'($signed(mixed_l))
                + 64'($signed(scale_g(sat_state(acc_l), GAIN_W'(SYMP_LEVEL_G)))));
        pre_r = sat_state(64'($signed(mixed_r))
                + 64'($signed(scale_g(sat_state(acc_r), GAIN_W'(SYMP_LEVEL_G)))));

        // -- room --------------------------------------------------------
        wet_l = '0;
        wet_r = '0;
        for (int c = 0; c < NUM_COMBS; c++) begin
          automatic int unsigned dl, dr;
          automatic logic signed [STATE_W-1:0] tl, tr, lpl, lpr;
          dl = (rev_ptr + REVERB_LEN - COMB_LEN[c]) % REVERB_LEN;
          dr = (rev_ptr + REVERB_LEN - COMB_LEN[c] - STEREO_OFFSET) % REVERB_LEN;
          tl = comb_mem_l[c][dl];
          tr = comb_mem_r[c][dr];
          // One-pole damper inside the feedback loop, so the tail loses
          // treble as it decays.  A tail that keeps its treble is a spring.
          lpl = sat_state(64'($signed(scale_g(tl, GAIN_W'(GAIN_W'(1 <<< GAIN_Q) - ROOM_DAMPING))))
                + 64'($signed(scale_g(comb_lp_l[c], GAIN_W'(ROOM_DAMPING)))));
          lpr = sat_state(64'($signed(scale_g(tr, GAIN_W'(GAIN_W'(1 <<< GAIN_Q) - ROOM_DAMPING))))
                + 64'($signed(scale_g(comb_lp_r[c], GAIN_W'(ROOM_DAMPING)))));
          comb_lp_l[c] <= lpl;
          comb_lp_r[c] <= lpr;
          comb_mem_l[c][rev_ptr] <= sat_state(64'($signed(pre_l))
                                    + 64'($signed(scale_g(lpl, GAIN_W'(ROOM_FEEDBACK)))));
          comb_mem_r[c][rev_ptr] <= sat_state(64'($signed(pre_r))
                                    + 64'($signed(scale_g(lpr, GAIN_W'(ROOM_FEEDBACK)))));
          wet_l = wet_l + 64'($signed(tl));
          wet_r = wet_r + 64'($signed(tr));
        end
        wet_l = wet_l / NUM_COMBS;
        wet_r = wet_r / NUM_COMBS;

        for (int a = 0; a < NUM_ALLPASS; a++) begin
          automatic int unsigned da;
          automatic logic signed [STATE_W-1:0] bufd, inp, outp;
          da   = (rev_ptr + REVERB_LEN - ALLPASS_LEN[a]) % REVERB_LEN;
          inp  = sat_state(wet_l);
          bufd = ap_mem_l[a][da];
          outp = sat_state(64'($signed(bufd)) - 64'($signed(scale_g(inp, GAIN_W'(GAIN_W'(1 <<< (GAIN_Q-1)))))));
          ap_mem_l[a][rev_ptr] <= sat_state(64'($signed(inp))
                                  + 64'($signed(scale_g(bufd, GAIN_W'(GAIN_W'(1 <<< (GAIN_Q-1)))))));
          wet_l = 64'($signed(outp));

          inp  = sat_state(wet_r);
          bufd = ap_mem_r[a][da];
          outp = sat_state(64'($signed(bufd)) - 64'($signed(scale_g(inp, GAIN_W'(GAIN_W'(1 <<< (GAIN_Q-1)))))));
          ap_mem_r[a][rev_ptr] <= sat_state(64'($signed(inp))
                                  + 64'($signed(scale_g(bufd, GAIN_W'(GAIN_W'(1 <<< (GAIN_Q-1)))))));
          wet_r = 64'($signed(outp));
        end

        // Wrap at the buffer length, not at the power of two the pointer
        // width implies: REVERB_LEN is 1784 while the pointer holds 2048, so a
        // plain increment walks off the end of every delay line and turns the
        // room into a comb resonance.
        rev_ptr <= (rev_ptr == ($clog2(REVERB_LEN))'(REVERB_LEN-1))
                   ? '0 : rev_ptr + 1'b1;

        // Wet is normalised by the network's own gain, not by the dry peak.
        begin
          automatic logic signed [STATE_W-1:0] w_l, w_r, final_l, final_r;
          w_l = scale_g(sat_state(wet_l), GAIN_W'(ROOM_WET_G));
          w_r = scale_g(sat_state(wet_r), GAIN_W'(ROOM_WET_G));
          final_l = sat_state(64'($signed(pre_l))
                    - 64'($signed(scale_g(pre_l, GAIN_W'(ROOM_MIX_G))))
                    + 64'($signed(scale_g(w_l,   GAIN_W'(ROOM_MIX_G)))));
          final_r = sat_state(64'($signed(pre_r))
                    - 64'($signed(scale_g(pre_r, GAIN_W'(ROOM_MIX_G))))
                    + 64'($signed(scale_g(w_r,   GAIN_W'(ROOM_MIX_G)))));
          // Standing headroom: the mix of a dense pedalled passage runs well
          // past full scale, and there is no look-ahead in hardware to
          // normalise it after the fact.
          out_l <= sat_pcm(64'($signed(scale_g(final_l, GAIN_W'(MASTER_GAIN_G))))
                           >>> (STATE_Q - 23));
          out_r <= sat_pcm(64'($signed(scale_g(final_r, GAIN_W'(MASTER_GAIN_G))))
                           >>> (STATE_Q - 23));
        end
        out_valid <= 1'b1;
      end
    end
  end

endmodule

`default_nettype wire
