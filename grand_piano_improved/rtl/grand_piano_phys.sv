// Piano physics, in SystemVerilog.
//
// This is `tools/piano_physics.py` rewritten as real-valued elaboration-time
// functions, so the RTL can build its own coefficient tables instead of
// reading 69,496 numbers out of generated .hex files.  Nothing here is
// hardware: every function runs once, before time zero, and the results are
// baked into the ROM arrays.  $sin, $cos, $exp, $ln, $sqrt and $pow are
// IEEE 1800 real math system functions.
//
// The equations are the same ones and in the same order as the Python, and
// `tools/check_rtl_coefs.py` compares the two tables entry by entry.  If you
// change a constant here, change it there.
`ifndef GRAND_PIANO_PHYS_SV
`define GRAND_PIANO_PHYS_SV

package grand_piano_phys;

  localparam real PI = 3.14159265358979323846;

  localparam int  PIANO_LOW_MIDI  = 21;
  localparam int  PIANO_HIGH_MIDI = 108;
  localparam real A4_HZ        = 440.0;
  localparam real A4_MIDI      = 69.0;

  // ------------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------------
  function automatic real clip(real x, real lo, real hi);
    clip = (x < lo) ? lo : (x > hi) ? hi : x;
  endfunction

  // Python's round() is half-to-even, and every table here is quantised with
  // it.  Ties are vanishingly unlikely among products of transcendentals, but
  // matching the rule costs four lines and removes a reason for the tables to
  // disagree.
  function automatic longint round_half_even(real x);
    real f, frac;
    longint i;
    f = $floor(x);
    i = longint'(f);
    frac = x - f;
    if (frac > 0.5) i = i + 1;
    else if (frac == 0.5 && (i % 2 != 0)) i = i + 1;
    round_half_even = i;
  endfunction

  function automatic longint to_fixed(real value, int frac, int width);
    longint v, lo, hi;
    v  = round_half_even(value * $pow(2.0, real'(frac)));
    hi = (longint'(1) <<< (width - 1)) - 1;
    lo = -(longint'(1) <<< (width - 1));
    to_fixed = (v < lo) ? lo : (v > hi) ? hi : v;
  endfunction

  // ------------------------------------------------------------------------
  // Partial phases -- SplitMix64, mirrored by hw_model.phase_table()
  // ------------------------------------------------------------------------
  function automatic real phase_at(longint unsigned seed, int index);
    longint unsigned state, z;
    state = seed;
    for (int i = 0; i <= index; i++) begin
      state = state + 64'h9E3779B97F4A7C15;
      z = state;
      z = (z ^ (z >> 30)) * 64'hBF58476D1CE4E5B9;
      z = (z ^ (z >> 27)) * 64'h94D049BB133111EB;
      z = z ^ (z >> 31);
    end
    // (z >> 11) * 2**-53 lands in [0,1); scale to a full turn.
    phase_at = real'(z >> 11) * $pow(2.0, -53.0) * 2.0 * PI;
  endfunction

  // ------------------------------------------------------------------------
  // Pitch and inharmonicity
  // ------------------------------------------------------------------------
  function automatic real fundamental(real midi);
    fundamental = A4_HZ * $pow(2.0, (midi - A4_MIDI) / 12.0);
  endfunction

  // Quadratic fit of log10(B) against MIDI number, through five concert-grand
  // anchor measurements.  Held as its fitted coefficients because a least
  // squares solve is not something to do at elaboration.
  localparam real B_FIT2 =  0.0005200751450972282;
  localparam real B_FIT1 = -0.045612909734755784;
  localparam real B_FIT0 = -3.2174976273587474;

  function automatic real inharmonicity(real midi);
    inharmonicity = $pow(10.0, (B_FIT2 * midi + B_FIT1) * midi + B_FIT0);
  endfunction

  function automatic real partial_freq(real midi, int n);
    real nn;
    nn = real'(n);
    partial_freq = nn * fundamental(midi)
                 * $sqrt(1.0 + inharmonicity(midi) * nn * nn);
  endfunction

  function automatic int partial_count(real midi, real sample_rate, int cap);
    real limit;
    int count, nxt;
    limit = 0.92 * sample_rate / 2.0;
    count = 1;
    while (count < cap) begin
      nxt = count + 1;
      if (partial_freq(midi, nxt) > limit) break;
      count = nxt;
    end
    partial_count = (count < 1) ? 1 : count;
  endfunction

  // ------------------------------------------------------------------------
  // Damping
  // ------------------------------------------------------------------------
  localparam real T60_REFERENCE_HZ = 100.0;
  localparam real T60_AT_REFERENCE = 24.0;
  localparam real T60_SLOPE        = 0.66;
  localparam real T60_FLOOR        = 0.40;
  localparam real T60_CEILING      = 40.0;

  function automatic real t60(real freq);
    real f;
    f = (freq < 1.0) ? 1.0 : freq;
    t60 = clip(T60_AT_REFERENCE * $pow(T60_REFERENCE_HZ / f, T60_SLOPE),
               T60_FLOOR, T60_CEILING);
  endfunction

  function automatic real decay_rate(real freq);
    decay_rate = $ln(1000.0) / t60(freq);
  endfunction

  localparam real PROMPT_LEVEL              = 0.45;
  localparam real AFTER_LEVEL               = 0.55;
  localparam real PROMPT_T60_SCALE          = 0.55;
  localparam real AFTER_T60_SCALE           = 1.0;
  localparam real POLARISATION_DETUNE_RATIO = 4.0e-4;

  // ------------------------------------------------------------------------
  // Unisons
  // ------------------------------------------------------------------------
  function automatic int strings_per_note(int midi);
    strings_per_note = (midi <= 27) ? 1 : (midi <= 40) ? 2 : 3;
  endfunction

  function automatic real unison_detune_cents(int midi, int index, int count);
    real spread, shape;
    if (count == 1) return 0.0;
    spread = 0.9 + 1.4 * clip((60.0 - real'(midi)) / 39.0, 0.0, 1.0e9);
    if (count == 2) begin
      unison_detune_cents = spread * ((index == 0) ? -0.5 : 0.5);
    end else begin
      shape = (index == 0) ? -0.5 : (index == 1) ? 0.12 : 0.55;
      unison_detune_cents = spread * shape;
    end
  endfunction

  // ------------------------------------------------------------------------
  // Hammer
  // ------------------------------------------------------------------------
  localparam real STRIKE_POSITION = 0.125;
  localparam real PARTIAL_SLOPE   = 1.35;
  localparam real COMB_FLOOR      = 0.20;

  function automatic real contact_corner_hz(real midi, real velocity);
    real v, pitch_term;
    v = clip(velocity / 127.0, 0.02, 1.0);
    pitch_term = 900.0 * $pow(2.0, (midi - 21.0) / 27.0);
    contact_corner_hz = pitch_term * (0.42 + 1.35 * $pow(v, 0.62));
  endfunction

  function automatic real contact_time_s(real midi, real velocity);
    real v, base;
    v = clip(velocity / 127.0, 0.02, 1.0);
    base = 0.0048 * $pow(2.0, -(midi - real'(PIANO_LOW_MIDI)) / 30.0);
    contact_time_s = base * (1.25 - 0.45 * v);
  endfunction

  function automatic real strike_position(real midi);
    strike_position = STRIKE_POSITION
                    * (1.0 - 0.30 * clip((midi - 60.0) / 48.0, 0.0, 1.0));
  endfunction

  // Excitation amplitude of one partial: strike-point comb, modal tilt, and
  // the hammer contact low-pass that makes soft playing dull.
  function automatic real partial_amplitude(real midi, real velocity, int n);
    real nn, comb, tilt, lowpass, v, f;
    nn      = real'(n);
    comb    = COMB_FLOOR + (1.0 - COMB_FLOOR)
            * ((($sin(nn * PI * strike_position(midi))) < 0.0)
               ? -$sin(nn * PI * strike_position(midi))
               :  $sin(nn * PI * strike_position(midi)));
    tilt    = $pow(nn, -PARTIAL_SLOPE);
    f       = partial_freq(midi, n);
    lowpass = 1.0 / (1.0 + $pow(f / contact_corner_hz(midi, velocity), 2.0));
    v       = clip(velocity / 127.0, 0.02, 1.0);
    partial_amplitude = comb * tilt * lowpass * $pow(v, 1.35);
  endfunction

  // ------------------------------------------------------------------------
  // Duplex / aliquot
  // ------------------------------------------------------------------------
  localparam int  DUPLEX_FROM_MIDI = 48;
  localparam real DUPLEX_T60       = 0.55;

  function automatic real duplex_ratio(int k);
    duplex_ratio = (k == 0) ? 8.02 : (k == 1) ? 12.05 : 16.10;
  endfunction

  function automatic real duplex_amp(real midi, real velocity, int k);
    real v, register, level;
    if (midi < real'(DUPLEX_FROM_MIDI)) return 0.0;
    v = clip(velocity / 127.0, 0.0, 1.0);
    register = clip((midi - real'(DUPLEX_FROM_MIDI)) / 40.0, 0.0, 1.0);
    level = 0.030 * (0.35 + 0.65 * register) * $pow(v, 1.3);
    duplex_amp = level / (1.0 + real'(k));
  endfunction

  // ------------------------------------------------------------------------
  // Dampers
  // ------------------------------------------------------------------------
  localparam real DAMPER_T60_BASS   = 0.42;
  localparam real DAMPER_T60_TREBLE = 0.11;
  localparam int  UNDAMPED_FROM_MIDI = 93;

  function automatic real damper_t60(int midi);   // <0 means "no damper"
    real x;
    if (midi >= UNDAMPED_FROM_MIDI) return -1.0;
    x = clip((real'(midi) - real'(PIANO_LOW_MIDI))
             / real'(UNDAMPED_FROM_MIDI - PIANO_LOW_MIDI), 0.0, 1.0);
    damper_t60 = DAMPER_T60_BASS + (DAMPER_T60_TREBLE - DAMPER_T60_BASS) * x;
  endfunction

  // ------------------------------------------------------------------------
  // Longitudinal ("phantom") partials
  // ------------------------------------------------------------------------
  localparam real LONGITUDINAL_SPEED          = 5047.544651250688;  // sqrt(E/rho)
  localparam real LONGITUDINAL_FROM_MIDI      = 45.0;
  localparam real LONGITUDINAL_LEVEL          = 0.055;
  localparam real LONGITUDINAL_VELOCITY_POWER = 3.5;
  localparam real LONGITUDINAL_T60_SCALE      = 0.18;

  // Speaking length, log-linear between eight anchor points.
  function automatic real speaking_length(real midi);
    real am [0:7];
    real al [0:7];
    real x, lo, hi, frac;
    am = '{21.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0};
    al = '{1.95, 1.30, 0.90, 0.62, 0.31, 0.15, 0.08, 0.05};
    x = clip(midi, am[0], am[7]);
    for (int i = 0; i < 7; i++) begin
      if (x >= am[i] && x <= am[i+1]) begin
        frac = (am[i+1] == am[i]) ? 0.0 : (x - am[i]) / (am[i+1] - am[i]);
        lo = $ln(al[i]);
        hi = $ln(al[i+1]);
        return $exp(lo + (hi - lo) * frac);
      end
    end
    speaking_length = al[7];
  endfunction

  function automatic real longitudinal_freq(real midi, int m);
    longitudinal_freq = LONGITUDINAL_SPEED / (2.0 * speaking_length(midi))
                      * real'(m);
  endfunction

  function automatic real longitudinal_amp(real midi, real velocity, int m);
    real v, register, level;
    if (midi > LONGITUDINAL_FROM_MIDI) return 0.0;
    v = clip(velocity / 127.0, 0.0, 1.0);
    register = clip((LONGITUDINAL_FROM_MIDI - midi) / 24.0, 0.0, 1.0);
    level = LONGITUDINAL_LEVEL * register * $pow(v, LONGITUDINAL_VELOCITY_POWER);
    if (level <= 0.0) return 0.0;
    longitudinal_amp = level / $pow(real'(m), 1.5);
  endfunction

  // ------------------------------------------------------------------------
  // Radiation
  // ------------------------------------------------------------------------
  localparam real RADIATION_CORNER_HZ  = 110.0;
  localparam real RADIATION_FLOOR      = 0.32;
  localparam real RADIATION_ROLLOFF_HZ = 5200.0;

  function automatic real radiation_gain(real freq);
    real f, ratio, rise, fall;
    f = (freq < 1.0) ? 1.0 : freq;
    ratio = f / RADIATION_CORNER_HZ;
    rise  = (ratio * ratio) / (1.0 + ratio * ratio);
    fall  = 1.0 / (1.0 + (f / RADIATION_ROLLOFF_HZ) * (f / RADIATION_ROLLOFF_HZ));
    radiation_gain = RADIATION_FLOOR + (1.0 - RADIATION_FLOOR) * rise * fall;
  endfunction

  // ------------------------------------------------------------------------
  // Stereo: a spaced microphone pair over the instrument
  // ------------------------------------------------------------------------
  localparam real SPEED_OF_SOUND = 343.0;
  localparam real PLATE_SPAN_X   = 1.40;
  localparam real PLATE_SPAN_Y   = 1.52;
  localparam real MIC_HEIGHT     = 0.55;
  localparam real MIC_LEFT_X     = -0.10, MIC_LEFT_Y  = 0.15;
  localparam real MIC_RIGHT_X    =  1.10, MIC_RIGHT_Y = 0.90;

  function automatic real mic_distance(int midi, bit right);
    real x, sx, sy, mx, my, dx, dy;
    x  = clip((real'(midi) - real'(PIANO_LOW_MIDI))
              / real'(PIANO_HIGH_MIDI - PIANO_LOW_MIDI), 0.0, 1.0);
    sx = (0.12 + 0.76 * x) * PLATE_SPAN_X;
    sy = (0.80 - 0.58 * x) * PLATE_SPAN_Y;
    mx = right ? MIC_RIGHT_X : MIC_LEFT_X;
    my = right ? MIC_RIGHT_Y : MIC_LEFT_Y;
    dx = sx - mx * PLATE_SPAN_X;
    dy = sy - my * PLATE_SPAN_Y;
    mic_distance = $sqrt(dx * dx + dy * dy + MIC_HEIGHT * MIC_HEIGHT);
  endfunction

  // ------------------------------------------------------------------------
  // Resonator coefficients -- the magic circle
  // ------------------------------------------------------------------------
  // k = 2*sin(theta/2) and loss = 1 - exp(-rate/fs), never cos(theta) and r.
  // At 27.5 Hz in Q1.16 both cos(theta) and r round to exactly 1.0, so the
  // bottom A would neither oscillate nor decay.
  function automatic real coef_k(real freq, real sample_rate);
    coef_k = 2.0 * $sin(PI * freq / sample_rate);
  endfunction

  function automatic real coef_loss(real rate, real sample_rate);
    coef_loss = 1.0 - $exp(-rate / sample_rate);
  endfunction

  // A resonator driven once per sample settles near g/(2*lambda), so scaling
  // the drive by 2*lambda normalises its peak.
  function automatic real driven_gain(real loss, real weight);
    driven_gain = 2.0 * loss * weight;
  endfunction

endpackage

`endif
