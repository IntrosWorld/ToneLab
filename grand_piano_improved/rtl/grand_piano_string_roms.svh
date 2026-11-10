// Builds the string coefficient tables at elaboration, from the physics in
// grand_piano_phys.sv, instead of reading them out of generated .hex files.
//
// Included inside grand_piano_core so the tasks can write the ROM arrays
// directly.  Everything here runs once, before time zero.
//
// The component order must match hw_model.components() exactly, because
// tools/check_rtl_coefs.py compares the two tables entry by entry.

  import grand_piano_phys::*;

  // Model sizes: the hardware carries fewer partials and strings than the
  // Python design model, which is what makes 88 notes fit in 8356 slots.
  localparam int HW_UNISON_PARTIALS = 24;
  localparam int HW_TWIN_PARTIALS   = 24;
  localparam int HW_MAX_STRINGS     = 2;
  localparam int HW_PARTIAL_CAP     = 52;
  localparam int DUPLEX_COUNT       = 3;

  // Per-note, per-velocity-layer output scaling.  This is the one string
  // quantity that is not computed here: it is note_loudness divided by the
  // peak of the summed note over its first 300 ms, and finding that peak
  // means summing every component across 14,400 samples -- 481 million
  // evaluations across the compass, which is fine in a compiled simulator and
  // hopeless in an interpreted elaborator.  352 numbers, against the 69,496
  // this file replaces.
  `include "grand_piano_norm.svh"

  function automatic real velocity_for_layer(int layer);
    velocity_for_layer = (real'(layer) + 0.5) * 127.0 / real'(VELOCITY_LAYERS);
  endfunction

  function automatic real note_loudness(int midi, real velocity);
    real v, bass, treble;
    v      = clip(velocity / 127.0, 0.02, 1.0);
    bass   = 1.0 + 0.35 * clip((52.0 - real'(midi)) / 31.0, 0.0, 1.0);
    treble = 1.0 - 0.24 * clip((real'(midi) - 52.0) / 56.0, 0.0, 1.0);
    note_loudness = 0.9 * $pow(v, 1.25) * bass * treble;
  endfunction

  // One note's component list, in hw_model.components() order.
  // kind: 0 = partial (prompt/after/plain), 1 = duplex, 2 = longitudinal.
  typedef struct {
    real freq;
    real rate;
    real phase;
    int  kind;
    int  index;
    real scale;
  } comp_t;

  function automatic int build_components(int pitch, ref comp_t list [0:SLOTS_PER_VOICE-1]);
    int count, strings, voices, n_out;
    real base, rate, freq, phase, scale, cents, f0, ghost;
    count   = partial_count(real'(pitch), real'(SAMPLE_RATE), 64);
    if (count > HW_PARTIAL_CAP) count = HW_PARTIAL_CAP;
    strings = strings_per_note(pitch);
    if (strings > HW_MAX_STRINGS) strings = HW_MAX_STRINGS;
    n_out = 0;

    for (int n = 0; n < count; n++) begin
      base   = partial_freq(real'(pitch), n + 1);
      rate   = decay_rate(base);
      voices = (n < HW_UNISON_PARTIALS) ? strings : 1;
      for (int s = 0; s < voices; s++) begin
        cents = (n < HW_UNISON_PARTIALS)
              ? unison_detune_cents(pitch, s, strings) : 0.0;
        freq  = base * $pow(2.0, cents / 1200.0);
        phase = phase_at(longint'(9000 + pitch), n) + real'(s) * 1.7;
        scale = 1.0 / $sqrt(real'(voices));
        if (n < HW_TWIN_PARTIALS) begin
          // The two transverse polarisations: a fast one that drives the
          // bridge and a slow aftersound, split slightly in frequency.
          if (n_out < SLOTS_PER_VOICE)
            list[n_out++] = '{freq, rate / PROMPT_T60_SCALE, phase, 0, n,
                              scale * PROMPT_LEVEL};
          if (n_out < SLOTS_PER_VOICE)
            list[n_out++] = '{freq * (1.0 + POLARISATION_DETUNE_RATIO),
                              rate * AFTER_T60_SCALE, phase + 0.9, 0, n,
                              scale * AFTER_LEVEL};
        end else begin
          if (n_out < SLOTS_PER_VOICE)
            list[n_out++] = '{freq, rate, phase, 0, n, scale};
        end
      end
    end

    f0 = fundamental(real'(pitch));
    for (int k = 0; k < DUPLEX_COUNT; k++) begin
      freq = f0 * duplex_ratio(k);
      if (pitch >= DUPLEX_FROM_MIDI && freq < 0.45 * real'(SAMPLE_RATE)
          && n_out < SLOTS_PER_VOICE)
        list[n_out++] = '{freq, $ln(1000.0) / DUPLEX_T60,
                          0.7 + 1.3 * real'(k), 1, k, 1.0};
    end

    // Frequencies always at full velocity; only the amplitude follows the blow.
    if (real'(pitch) <= LONGITUDINAL_FROM_MIDI
        && longitudinal_amp(real'(pitch), 127.0, 1) > 0.0) begin
      for (int m = 1; m <= 3; m++) begin
        ghost = longitudinal_freq(real'(pitch), m);
        if (ghost < 0.45 * real'(SAMPLE_RATE) && n_out < SLOTS_PER_VOICE)
          list[n_out++] = '{ghost,
                            decay_rate(ghost) / LONGITUDINAL_T60_SCALE,
                            0.31, 2, m - 1, 1.0};
      end
    end

    build_components = n_out;
  endfunction

  // Amplitude of one component at one velocity, before loudness normalisation.
  function automatic real component_amplitude(int pitch, real velocity,
                                              comp_t c);
    case (c.kind)
      0: component_amplitude =
             partial_amplitude(real'(pitch), velocity, c.index + 1)
           * radiation_gain(partial_freq(real'(pitch), c.index + 1))
           * c.scale;
      1: component_amplitude = duplex_amp(real'(pitch), velocity, c.index);
      default: component_amplitude =
             longitudinal_amp(real'(pitch), velocity, c.index + 1);
    endcase
  endfunction

  task automatic build_string_roms();
    comp_t list [0:SLOTS_PER_VOICE-1];
    int n_comp, entry, midi;
    real amp, norm, tau, t60_d;
    entry = 0;
    for (int note = 0; note < NUM_NOTES; note++) begin
      midi   = LOWEST_MIDI + note;
      n_comp = build_components(midi, list);
      rom_base[note]  = entry;
      rom_count[note] = n_comp;

      for (int i = 0; i < n_comp; i++) begin
        rom_k[entry]    = COEF_W'(to_fixed(coef_k(list[i].freq,
                                                  real'(SAMPLE_RATE)),
                                           COEF_Q, COEF_W));
        rom_loss[entry] = LOSS_W'(to_fixed(coef_loss(list[i].rate,
                                                     real'(SAMPLE_RATE)),
                                           LOSS_Q, LOSS_W));
        rom_cphi[entry] = COEF_W'(to_fixed($cos(list[i].phase), COEF_Q, COEF_W));
        rom_sphi[entry] = COEF_W'(to_fixed($sin(list[i].phase), COEF_Q, COEF_W));

        for (int v = 0; v < VELOCITY_LAYERS; v++) begin
          norm = NOTE_NORM[note][v];
          amp  = component_amplitude(midi, velocity_for_layer(v), list[i]) * norm;
          case (v)
            0: rom_amp0[entry] = STATE_W'(to_fixed(amp, STATE_Q, STATE_W));
            1: rom_amp1[entry] = STATE_W'(to_fixed(amp, STATE_Q, STATE_W));
            2: rom_amp2[entry] = STATE_W'(to_fixed(amp, STATE_Q, STATE_W));
            default: rom_amp3[entry] = STATE_W'(to_fixed(amp, STATE_Q, STATE_W));
          endcase
        end
        entry++;
      end

      // Hammer contact ramp: the attack is a one-pole rise, not a step.  This
      // is what separates a struck string from a plucked one.
      tau = contact_time_s(real'(midi), 100.0) / 3.0;
      rom_attack[note] = GAIN_W'(to_fixed(
          1.0 - $exp(-1.0 / (tau * real'(SAMPLE_RATE))), GAIN_Q, GAIN_W));

      t60_d = damper_t60(midi);
      if (t60_d < 0.0) begin
        rom_damper[note]   = GAIN_W'(to_fixed(1.0, GAIN_Q, GAIN_W));
        rom_undamped[note] = 8'd1;
      end else begin
        rom_damper[note] = GAIN_W'(to_fixed(
            $exp(-$ln(1000.0) / (t60_d * real'(SAMPLE_RATE))), GAIN_Q, GAIN_W));
        rom_undamped[note] = 8'd0;
      end
    end
  endtask

  // Stereo is arrival geometry, not level panning: a spaced pair over the
  // instrument hears each string at a different time.  Only the difference
  // between the two arrivals matters, so the common part is removed -- which
  // is what turns a 200-sample buffer into a 64-sample one.
  task automatic build_stereo_rom();
    real dl [0:NUM_NOTES-1];
    real dr [0:NUM_NOTES-1];
    real floor_s, near;
    for (int note = 0; note < NUM_NOTES; note++) begin
      dl[note] = mic_distance(LOWEST_MIDI + note, 1'b0);
      dr[note] = mic_distance(LOWEST_MIDI + note, 1'b1);
    end
    floor_s = 1.0e30;
    for (int note = 0; note < NUM_NOTES; note++) begin
      near = (dl[note] < dr[note]) ? dl[note] : dr[note];
      if (near / SPEED_OF_SOUND * real'(SAMPLE_RATE) < floor_s)
        floor_s = near / SPEED_OF_SOUND * real'(SAMPLE_RATE);
    end
    for (int note = 0; note < NUM_NOTES; note++) begin
      near = (dl[note] < dr[note]) ? dl[note] : dr[note];
      rom_delay_l[note] = 16'(round_half_even(
          dl[note] / SPEED_OF_SOUND * real'(SAMPLE_RATE) - floor_s));
      rom_delay_r[note] = 16'(round_half_even(
          dr[note] / SPEED_OF_SOUND * real'(SAMPLE_RATE) - floor_s));
      rom_pan_l[note] = GAIN_W'(to_fixed(near / dl[note], GAIN_Q, GAIN_W));
      rom_pan_r[note] = GAIN_W'(to_fixed(near / dr[note], GAIN_Q, GAIN_W));
    end
  endtask
