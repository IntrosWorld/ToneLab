// Builds the sympathetic-string table at elaboration, and holds the
// soundboard modes as literals.
//
// Included inside grand_piano_fx so the tasks can write its arrays directly.
//
// The soundboard is the one part of the instrument that is not a closed-form
// equation.  Its modes come from a finite-element solve of an orthotropic
// spruce plate -- assemble the stiffness and mass matrices, then a sparse
// shift-invert eigensolve.  That is not something to attempt at elaboration,
// so the 64 resulting numbers are quantised once and written here.  They are
// the only part of the instrument the RTL cannot derive for itself.

  import grand_piano_phys::*;

  `include "grand_piano_body.svh"

  localparam real SYMP_MULT   [0:2] = '{1.0, 2.0, 3.0};
  localparam real SYMP_WEIGHT [0:2] = '{1.0, 0.4, 0.18};

  task automatic build_body_rom();
    for (int i = 0; i < BODY_MODES; i++) begin
      body_k[i]    = COEF_W'(BODY_K_Q[i]);
      body_loss[i] = LOSS_W'(BODY_LOSS_Q[i]);
      body_gl[i]   = COEF_W'(BODY_GL_Q[i]);
      body_gr[i]   = COEF_W'(BODY_GR_Q[i]);
    end
  endtask

  // Strings that are free to ring: the pedalled ones, plus the top octave,
  // which has no damper at all.  Each is driven by the mix, so its gain is
  // scaled by 2*loss to normalise the peak of a once-per-sample drive.
  task automatic build_symp_rom();
    int slot;
    real f0, freq, rate, loss, gain, gl, gr, near, dl, dr;
    bit undamped;
    slot = 0;
    for (int pitch = 24; pitch <= PIANO_HIGH_MIDI; pitch++) begin
      f0 = fundamental(real'(pitch));
      if (f0 > 4500.0) break;
      undamped = (pitch >= UNDAMPED_FROM_MIDI);

      dl   = mic_distance(pitch, 1'b0);
      dr   = mic_distance(pitch, 1'b1);
      near = (dl < dr) ? dl : dr;
      gl   = near / dl;
      gr   = near / dr;

      for (int m = 0; m < 3; m++) begin
        freq = f0 * SYMP_MULT[m];
        if (freq >= 0.45 * real'(SAMPLE_RATE) || slot >= SYMP_SLOTS) continue;
        rate = decay_rate(freq);
        loss = coef_loss(rate, real'(SAMPLE_RATE));
        gain = driven_gain(loss,
                           SYMP_WEIGHT[m] * (undamped ? 0.55 : 1.0));
        symp_k[slot]     = COEF_W'(to_fixed(coef_k(freq, real'(SAMPLE_RATE)),
                                            COEF_Q, COEF_W));
        symp_loss[slot]  = LOSS_W'(to_fixed(loss, LOSS_Q, LOSS_W));
        symp_gain[slot]  = COEF_W'(to_fixed(gain, COEF_Q, COEF_W));
        symp_pl[slot]    = GAIN_W'(to_fixed(gl, GAIN_Q, GAIN_W));
        symp_pr[slot]    = GAIN_W'(to_fixed(gr, GAIN_Q, GAIN_W));
        symp_und[slot]   = undamped ? 8'd1 : 8'd0;
        symp_pitch[slot] = 8'(pitch);
        slot++;
      end
    end
    while (slot < SYMP_SLOTS) begin
      symp_k[slot]     = '0;
      symp_loss[slot]  = '0;
      symp_gain[slot]  = '0;
      symp_pl[slot]    = '0;
      symp_pr[slot]    = '0;
      symp_und[slot]   = 8'd0;
      symp_pitch[slot] = 8'd0;
      slot++;
    end
  endtask
