// Grand piano string engine.
//
// A bank of TOTAL_SLOTS damped resonators, time-multiplexed through one
// datapath and summed.  Every element of the sound -- partials, the two string
// polarisations, the detuned unisons, the duplex segments beyond the bridge,
// and the longitudinal modes -- is one resonator entry loaded from ROM at
// note-on.  See tools/hw_model.py for how the ROM is built.
//
// Resonator form
// --------------
// Each slot runs the "magic circle" recurrence
//
//     x <- x + k*y
//     y <- y - k*x        (with the updated x)
//
// followed by an explicit loss
//
//     x <- x - lambda*x
//     y <- y - lambda*y
//
// which together give a*exp(-lambda*n)*sin(theta*n + phi) with
// theta = 2*asin(k/2).
//
// This form is chosen over the obvious direct-form biquad because of fixed
// point, not elegance.  A direct form stores cos(theta) and r = exp(-rate/fs),
// both of which sit just under 1.0 for a piano: at 27.5 Hz, cos(theta) rounds
// to exactly 1.0 in Q1.16 and the note never oscillates at all, while r rounds
// to 1.0 and it never decays.  Here the stored quantities k and lambda are
// themselves small, so a fixed-point word spends its bits on precision that
// matters.  Measured over the compass, this representation holds pitch to
// better than 0.0005 cents and T60 to better than 0.01%.
//
// The hammer contact ramp and the damper are per-voice gains rather than
// per-resonator state.  That is exact, not an approximation: both multiply
// every component of a note by the same factor, so they factor out of the sum.

`default_nettype none

module grand_piano_core
  import grand_piano_pkg::*;
#(
  // Clock cycles per audio sample.  100 MHz / 48 kHz = 2083, which must cover
  // one pass over every slot plus the worst-case note load.
  parameter int SAMPLE_PERIOD = 2083
) (
  input  wire                     clk,
  input  wire                     rst,

  // Note events.  Pulse valid for one cycle.
  input  wire                     note_on_valid,
  input  wire [6:0]               note_on_pitch,
  input  wire [6:0]               note_on_velocity,
  input  wire                     note_off_valid,
  input  wire [6:0]               note_off_pitch,
  input  wire                     sustain_pedal,

  // Check ready before presenting a nonzero-velocity note-on. Overflow is
  // sticky until reset, so hosts can detect a missed event instead of hanging
  // a note silently. Note-offs and pedal changes never need backpressure.
  output wire                     note_on_ready,
  output logic                    event_overflow,

  output logic                    sample_valid,
  output logic signed [STATE_W-1:0] dry_l,
  output logic signed [STATE_W-1:0] dry_r,

  // One bit per MIDI pitch, high while a voice is sounding it.  The
  // sympathetic bank uses this to avoid driving a string with itself: that
  // string is already ringing as a struck note, and letting it also resonate
  // adds a second, longer copy of the same pitch.
  output logic [127:0]            pitch_active
);

  localparam int SLOT_BITS  = $clog2(TOTAL_SLOTS);
  localparam int VOICE_BITS = $clog2(NUM_VOICES);
  localparam int LOCAL_BITS = $clog2(SLOTS_PER_VOICE);

  // ---------------------------------------------------------------------
  // Coefficient ROM
  // ---------------------------------------------------------------------
  logic signed [COEF_W-1:0]  rom_k     [ROM_ENTRIES];
  logic signed [LOSS_W-1:0]  rom_loss  [ROM_ENTRIES];
  logic signed [COEF_W-1:0]  rom_cphi  [ROM_ENTRIES];
  logic signed [COEF_W-1:0]  rom_sphi  [ROM_ENTRIES];
  logic signed [STATE_W-1:0] rom_amp0  [ROM_ENTRIES];
  logic signed [STATE_W-1:0] rom_amp1  [ROM_ENTRIES];
  logic signed [STATE_W-1:0] rom_amp2  [ROM_ENTRIES];
  logic signed [STATE_W-1:0] rom_amp3  [ROM_ENTRIES];
  logic        [31:0]        rom_base  [NUM_NOTES];
  logic        [31:0]        rom_count [NUM_NOTES];
  logic signed [GAIN_W-1:0]  rom_attack[NUM_NOTES];
  logic signed [GAIN_W-1:0]  rom_damper[NUM_NOTES];
  logic        [7:0]         rom_undamped[NUM_NOTES];
  logic        [15:0]        rom_delay_l [NUM_NOTES];
  logic        [15:0]        rom_delay_r [NUM_NOTES];
  logic signed [GAIN_W-1:0]  rom_pan_l   [NUM_NOTES];
  logic signed [GAIN_W-1:0]  rom_pan_r   [NUM_NOTES];

  // The tables are computed here, at elaboration, from the physics in
  // grand_piano_phys.sv -- not read from generated files.  All 67,640 string
  // values are bit-identical to the Python model; tools/check_rtl_coefs.py
  // compares them entry by entry.
  `include "grand_piano_string_roms.svh"

  initial begin
    if (SAMPLE_PERIOD < TOTAL_SLOTS/2 + SLOTS_PER_VOICE + 1)
      $fatal(1, "SAMPLE_PERIOD cannot cover a resonator pass and note load");
    if (SLOTS_PER_VOICE < 2 || (SLOTS_PER_VOICE & (SLOTS_PER_VOICE-1)) != 0)
      $fatal(1, "SLOTS_PER_VOICE must be a power of two, at least two");
    if ((DELAY_LEN & (DELAY_LEN-1)) != 0)
      $fatal(1, "DELAY_LEN must be a power of two");
    build_string_roms();
    build_stereo_rom();
  end

  // ---------------------------------------------------------------------
  // Resonator state
  // ---------------------------------------------------------------------
  logic signed [STATE_W-1:0] slot_x    [TOTAL_SLOTS];
  logic signed [STATE_W-1:0] slot_y    [TOTAL_SLOTS];
  logic signed [COEF_W-1:0]  slot_k    [TOTAL_SLOTS];
  logic signed [LOSS_W-1:0]  slot_loss [TOTAL_SLOTS];

  // ---------------------------------------------------------------------
  // Voice bookkeeping
  // ---------------------------------------------------------------------
  logic                     v_active   [NUM_VOICES];
  logic                     v_released [NUM_VOICES];
  // Whether the player is still holding this key.  Needed to tell apart the
  // two reasons a note is sounding: the key is down, or the pedal is holding
  // it.  Without it, lifting the pedal cuts notes the player is still holding.
  logic                     v_key_down [NUM_VOICES];
  logic                     v_undamped [NUM_VOICES];
  logic [6:0]               v_pitch    [NUM_VOICES];
  logic signed [GAIN_W-1:0] v_attack   [NUM_VOICES];  // one-pole ramp rate
  logic signed [GAIN_W-1:0] v_env      [NUM_VOICES];  // contact ramp, rises to 1
  logic signed [GAIN_W-1:0] v_damper   [NUM_VOICES];  // per-sample decay once released
  logic signed [GAIN_W-1:0] v_gain     [NUM_VOICES];  // damper gain, starts at 1
  logic [31:0]              v_age      [NUM_VOICES];
  // A peak-follower on each voice's own output.  Stealing overwrites a voice's
  // resonators outright, so whatever it was sounding stops dead -- a step
  // discontinuity, which is a click.  demo-6 has 193 notes against 28 voices
  // and steals about 165 of them, and that broadband click energy is what
  // congests the midrange and buries the notes.  Taking the quietest voice
  // instead of the oldest makes each discontinuity as small as it can be.
  logic signed [STATE_W-1:0] v_level   [NUM_VOICES];
  logic [15:0]              v_delay_l  [NUM_VOICES];
  logic [15:0]              v_delay_r  [NUM_VOICES];
  logic signed [GAIN_W-1:0] v_pan_l    [NUM_VOICES];
  logic signed [GAIN_W-1:0] v_pan_r    [NUM_VOICES];

  // Notes are scattered across the board, so each reaches the two microphones
  // at a different time.  Rather than a delay line per voice, every voice adds
  // into a shared circular buffer at its own offset -- the arrival spread is
  // only a few dozen samples once the common part is removed.
  logic signed [STATE_W-1:0] delay_l [DELAY_LEN];
  logic signed [STATE_W-1:0] delay_r [DELAY_LEN];
  logic [$clog2(DELAY_LEN)-1:0] delay_wp;

  localparam logic signed [GAIN_W-1:0] GAIN_ONE = GAIN_W'(1) <<< GAIN_Q;

  // ---------------------------------------------------------------------
  // Control
  // ---------------------------------------------------------------------
  typedef enum logic [1:0] { ST_STEP, ST_LOAD, ST_WAIT } state_t;
  state_t state;

  logic [SLOT_BITS-1:0]  step_idx;
  logic [LOCAL_BITS:0]   load_idx;
  logic [31:0]           period_ctr;

  // Note-on queue.  The core can start one note per audio sample, but a chord
  // arrives as several note-ons on consecutive clocks -- without a queue every
  // note but the first is silently dropped, which is what removed the bass
  // from chords entirely.
  localparam int NOTE_FIFO_DEPTH = 32;
  logic [6:0] fifo_pitch [NOTE_FIFO_DEPTH];
  logic [6:0] fifo_vel   [NOTE_FIFO_DEPTH];
  // A note-off can arrive before its note-on has been loaded -- the core only
  // starts one note per audio sample, so a queued note has no voice yet and
  // nothing for the note-off to match.  Such a note would then be loaded with
  // its key still down and never damp at all.  Marking the queue entry carries
  // the release across to the load.
  logic       fifo_off   [NOTE_FIFO_DEPTH];
  logic [$clog2(NOTE_FIFO_DEPTH):0] fifo_count;
  logic [$clog2(NOTE_FIFO_DEPTH)-1:0] fifo_head, fifo_tail;

  wire fifo_pop = (state == ST_STEP)
                 && (step_idx == SLOT_BITS'(TOTAL_SLOTS-2)) && (fifo_count != 0);
  wire valid_note_on = note_on_valid && (note_on_velocity != 0)
                      && (note_on_pitch >= 7'(LOWEST_MIDI))
                      && (note_on_pitch < 7'(LOWEST_MIDI + NUM_NOTES));
  assign note_on_ready = !rst && ((fifo_count < ($clog2(NOTE_FIFO_DEPTH)+1)'(NOTE_FIFO_DEPTH))
                                  || fifo_pop);
  wire fifo_push = valid_note_on && note_on_ready;

  function automatic logic matches_note_off(input logic [6:0] pitch);
    return (note_off_valid && note_off_pitch == pitch)
           || (note_on_valid && note_on_velocity == 0 && note_on_pitch == pitch);
  endfunction

  logic                  pending;
  logic                  pending_off;   // its note-off arrived before it loaded
  logic [6:0]            pending_pitch;
  logic [6:0]            pending_velocity;
  logic [VOICE_BITS-1:0] load_voice;
  logic [31:0]           load_base;
  logic [31:0]           load_count;
  logic [1:0]            load_layer;

  logic signed [63:0]    voice_acc;

  // Pedal held notes stay undamped until it lifts.
  logic sustain_q;
  // Dampers are felt masses on levers, not switches: they take tens of
  // milliseconds to reach the strings.  Pianists re-pedal constantly -- this
  // piece releases and re-presses within 1.92 ms -- and treating that as
  // instant damping silences every sounding note.  The dampers only land if
  // the pedal stays up long enough for them to fall.
  logic [$clog2(DAMPER_TRAVEL+1)-1:0] damper_fall;

  always_comb begin
    pitch_active = '0;
    for (int v = 0; v < NUM_VOICES; v++)
      if (v_active[v]) pitch_active[v_pitch[v]] = 1'b1;
  end

  // One pulse per audio sample.
  logic sample_tick;
  assign sample_tick = (state == ST_WAIT) && (period_ctr >= 32'(SAMPLE_PERIOD-1));

  // One resonator step: magic-circle rotation followed by the explicit loss.
  function automatic logic [2*STATE_W-1:0] step_slot(
      input logic signed [STATE_W-1:0] x,
      input logic signed [STATE_W-1:0] y,
      input logic signed [COEF_W-1:0]  k,
      input logic signed [LOSS_W-1:0]  loss);
    logic signed [STATE_W-1:0] xr, yr, xn, yn;
    xr = sat_state(64'($signed(x)) + (($signed(k) * $signed(y)) >>> COEF_Q));
    yr = sat_state(64'($signed(y)) - (($signed(k) * $signed(xr)) >>> COEF_Q));
    xn = sat_state(64'($signed(xr)) - (($signed(loss) * $signed(xr)) >>> LOSS_Q));
    yn = sat_state(64'($signed(yr)) - (($signed(loss) * $signed(yr)) >>> LOSS_Q));
    step_slot = {xn, yn};
  endfunction

  // ---------------------------------------------------------------------
  // Voice allocation
  // ---------------------------------------------------------------------
  // Free voice first, then a released one, then the quietest sounding voice.
  function automatic logic [VOICE_BITS-1:0] pick_voice();
    logic [VOICE_BITS-1:0] best;
    for (int v = 0; v < NUM_VOICES; v++)
      if (!v_active[v]) return VOICE_BITS'(v);
    for (int v = 0; v < NUM_VOICES; v++)
      if (v_released[v]) return VOICE_BITS'(v);
    best = '0;
    // Quietest sounding voice: its interruption is the least audible.
    for (int v = 0; v < NUM_VOICES; v++) begin
      if (v == 0 || $signed(v_level[v]) < $signed(v_level[best])) begin
        best = VOICE_BITS'(v);
      end
    end
    return best;
  endfunction

  always_ff @(posedge clk) begin
    if (rst) begin
      state        <= ST_STEP;
      step_idx     <= '0;
      load_idx     <= '0;
      period_ctr   <= '0;
      pending      <= 1'b0;
      pending_off  <= 1'b0;
      fifo_count   <= '0;
      fifo_head    <= '0;
      fifo_tail    <= '0;
      event_overflow <= 1'b0;
      sample_valid <= 1'b0;
      dry_l        <= '0;
      dry_r        <= '0;
      delay_wp     <= '0;
      voice_acc    <= '0;
      sustain_q    <= 1'b0;
      damper_fall  <= '0;
      for (int v = 0; v < NUM_VOICES; v++) begin
        v_active[v]   <= 1'b0;
        v_released[v] <= 1'b0;
        v_key_down[v] <= 1'b0;
        v_undamped[v] <= 1'b0;
        v_env[v]      <= '0;
        v_gain[v]     <= GAIN_ONE;
        v_age[v]      <= '0;
        v_level[v]    <= '0;
        v_pitch[v]    <= '0;
        v_attack[v]   <= '0;
        v_damper[v]   <= GAIN_ONE;
        v_delay_l[v]  <= '0;
        v_delay_r[v]  <= '0;
        v_pan_l[v]    <= '0;
        v_pan_r[v]    <= '0;
      end
      for (int d = 0; d < DELAY_LEN; d++) begin
        delay_l[d] <= '0;
        delay_r[d] <= '0;
      end
      for (int s = 0; s < TOTAL_SLOTS; s++) begin
        slot_x[s]    <= '0;
        slot_y[s]    <= '0;
        slot_k[s]    <= '0;
        slot_loss[s] <= '0;
      end
    end else begin
      sample_valid <= 1'b0;
      sustain_q    <= sustain_pedal;
      // Count the entire sample period, including computation and loading.
      // Counting only ST_WAIT makes cadence depend on whether a note starts.
      period_ctr   <= period_ctr + 1'b1;

      // -- event capture -------------------------------------------------
      if (valid_note_on && !note_on_ready) event_overflow <= 1'b1;
      case ({fifo_push, fifo_pop})
        2'b10: fifo_count <= fifo_count + 1'b1;
        2'b01: fifo_count <= fifo_count - 1'b1;
        default: ;
      endcase
      if (fifo_push) begin
        fifo_pitch[fifo_tail] <= note_on_pitch;
        fifo_vel  [fifo_tail] <= note_on_velocity;
        fifo_off  [fifo_tail] <= matches_note_off(note_on_pitch);
        fifo_tail  <= fifo_tail + 1'b1;
      end

      if (note_off_valid || (note_on_valid && note_on_velocity == 0)) begin
        // Only occupied entries may be marked. Exclude a same-cycle write:
        // its release flag is computed from the new pitch above.
        for (int q = 0; q < NOTE_FIFO_DEPTH; q++)
          if (($clog2(NOTE_FIFO_DEPTH))'(q - int'(fifo_head)) < fifo_count
              && !(fifo_push && q == int'(fifo_tail))
              && matches_note_off(fifo_pitch[q])) fifo_off[q] <= 1'b1;
        if (pending && matches_note_off(pending_pitch)) pending_off <= 1'b1;
        for (int v = 0; v < NUM_VOICES; v++) begin
          if (v_active[v] && matches_note_off(v_pitch[v])) begin
            v_key_down[v] <= 1'b0;
            // With the pedal down the damper never lands; the top strings
            // have no damper to land at all.
            if (!sustain_pedal && !v_undamped[v] && !v_released[v])
              v_released[v] <= 1'b1;
          end
        end
      end

      // Pedal up: let the dampers fall.  They only reach the strings once the
      // pedal has stayed up for DAMPER_TRAVEL samples; a quick re-pedal
      // catches them before they land and the notes ring on.  Notes whose key
      // is still down keep sounding either way -- lifting the pedal does not
      // stop the player's fingers.
      // DAMPER_TRAVEL is counted in audio samples, so it only advances on the
      // sample tick -- a per-clock count would fire 2083 times too fast.
      if (!sustain_pedal) begin
        if (sample_tick && damper_fall < ($clog2(DAMPER_TRAVEL+1))'(DAMPER_TRAVEL)) begin
          damper_fall <= damper_fall + 1'b1;
        end
        if (sample_tick && damper_fall == ($clog2(DAMPER_TRAVEL+1))'(DAMPER_TRAVEL-1)) begin
          for (int v = 0; v < NUM_VOICES; v++) begin
            if (v_active[v] && !v_undamped[v] && !v_key_down[v])
              v_released[v] <= 1'b1;
          end
        end
      end else begin
        damper_fall <= '0;
      end

      unique case (state)
        // -- one pass over every resonator ------------------------------
        ST_STEP: begin
          // Two slots per clock: a full pass then costs TOTAL_SLOTS/2 cycles,
          // which is what lets 28 voices fit inside the 2083-cycle budget.
          automatic logic [SLOT_BITS-1:0] idx_a, idx_b;
          automatic logic [2*STATE_W-1:0] res_a, res_b;
          automatic logic signed [STATE_W-1:0] yn;

          idx_a = step_idx;
          idx_b = step_idx + 1'b1;

          res_a = step_slot(slot_x[idx_a], slot_y[idx_a], slot_k[idx_a], slot_loss[idx_a]);
          res_b = step_slot(slot_x[idx_b], slot_y[idx_b], slot_k[idx_b], slot_loss[idx_b]);

          slot_x[idx_a] <= res_a[2*STATE_W-1:STATE_W];
          slot_y[idx_a] <= res_a[STATE_W-1:0];
          slot_x[idx_b] <= res_b[2*STATE_W-1:STATE_W];
          slot_y[idx_b] <= res_b[STATE_W-1:0];

          yn = sat_state(64'($signed(res_a[STATE_W-1:0]))
                         + 64'($signed(res_b[STATE_W-1:0])));
          voice_acc <= voice_acc + 64'($signed(yn));

          // Voice boundary: apply that voice's envelope and place the note.
          if (idx_b[LOCAL_BITS-1:0] == LOCAL_BITS'(SLOTS_PER_VOICE-1)) begin
            automatic logic [VOICE_BITS-1:0] vsel;
            automatic logic signed [63:0] scaled, enveloped;
            automatic logic [$clog2(DELAY_LEN)-1:0] wl, wr;
            vsel      = idx_b[SLOT_BITS-1:LOCAL_BITS];
            scaled    = ((voice_acc + 64'(yn)) * 64'($signed(v_env[vsel]))) >>> GAIN_Q;
            enveloped = (scaled * 64'($signed(v_gain[vsel]))) >>> GAIN_Q;

            wl = ($clog2(DELAY_LEN))'(delay_wp + v_delay_l[vsel][$clog2(DELAY_LEN)-1:0]);
            wr = ($clog2(DELAY_LEN))'(delay_wp + v_delay_r[vsel][$clog2(DELAY_LEN)-1:0]);
            delay_l[wl] <= sat_state(64'($signed(delay_l[wl]))
                           + ((enveloped * 64'($signed(v_pan_l[vsel]))) >>> GAIN_Q));
            delay_r[wr] <= sat_state(64'($signed(delay_r[wr]))
                           + ((enveloped * 64'($signed(v_pan_r[vsel]))) >>> GAIN_Q));

            // Peak-follow this voice, with a slow release, so the allocator
            // can tell which voice is quietest.
            begin
              automatic logic signed [63:0] mag;
              mag = enveloped[63] ? -enveloped : enveloped;
              if (mag > 64'($signed(v_level[vsel])))
                v_level[vsel] <= sat_state(mag);
              else
                v_level[vsel] <= v_level[vsel]
                                 - ($signed(v_level[vsel]) >>> 10);
            end
            voice_acc <= '0;
          end

          if (idx_b >= SLOT_BITS'(TOTAL_SLOTS-1)) begin
            step_idx <= '0;
            state    <= (fifo_count != 0) ? ST_LOAD : ST_WAIT;
            if (fifo_count != 0) begin
              automatic logic [6:0] p_pitch, p_vel;
              p_pitch = fifo_pitch[fifo_head];
              p_vel   = fifo_vel  [fifo_head];
              pending          <= 1'b1;
              pending_pitch    <= p_pitch;
              pending_velocity <= p_vel;
              pending_off      <= fifo_off[fifo_head] || matches_note_off(p_pitch);
              fifo_head  <= fifo_head + 1'b1;
              load_voice <= pick_voice();
              load_base  <= rom_base [p_pitch - 7'(LOWEST_MIDI)];
              load_count <= rom_count[p_pitch - 7'(LOWEST_MIDI)];
              load_layer <= p_vel[6:5];
              load_idx   <= '0;
            end
          end else begin
            step_idx <= step_idx + 2'd2;
          end
        end

        // -- load one note's resonators ---------------------------------
        ST_LOAD: begin
          automatic logic [SLOT_BITS-1:0] slot;
          automatic logic [31:0]          src;
          automatic logic signed [STATE_W-1:0] amp;
          automatic logic signed [63:0]   ix, iy;

          slot = {load_voice, load_idx[LOCAL_BITS-1:0]};
          src  = load_base + 32'(load_idx);

          if (load_idx < load_count[LOCAL_BITS:0]) begin
            unique case (load_layer)
              2'd0: amp = rom_amp0[src];
              2'd1: amp = rom_amp1[src];
              2'd2: amp = rom_amp2[src];
              default: amp = rom_amp3[src];
            endcase
            ix = 64'($signed(amp)) * 64'($signed(rom_cphi[src]));
            iy = 64'($signed(amp)) * 64'($signed(rom_sphi[src]));
            slot_x[slot]    <= sat_state(ix >>> COEF_Q);
            slot_y[slot]    <= sat_state(iy >>> COEF_Q);
            slot_k[slot]    <= rom_k[src];
            slot_loss[slot] <= rom_loss[src];
          end else begin
            // Silence the unused slots of this voice.
            slot_x[slot]    <= '0;
            slot_y[slot]    <= '0;
            slot_k[slot]    <= '0;
            slot_loss[slot] <= '0;
          end

          if (load_idx == (LOCAL_BITS+1)'(SLOTS_PER_VOICE-1)) begin
            automatic logic [6:0] idx;
            automatic logic released_before_load;
            idx = pending_pitch - 7'(LOWEST_MIDI);
            released_before_load = pending_off || matches_note_off(pending_pitch);
            v_active  [load_voice] <= 1'b1;
            // If the note-off already came and went while this note waited in
            // the queue, it starts already released rather than ringing on.
            v_released[load_voice] <= released_before_load && !sustain_pedal
                                      && !rom_undamped[idx][0];
            v_key_down[load_voice] <= !released_before_load;
            v_undamped[load_voice] <= rom_undamped[idx][0];
            v_pitch   [load_voice] <= pending_pitch;
            v_attack  [load_voice] <= rom_attack[idx];
            v_delay_l [load_voice] <= rom_delay_l[idx];
            v_delay_r [load_voice] <= rom_delay_r[idx];
            v_pan_l   [load_voice] <= rom_pan_l[idx];
            v_pan_r   [load_voice] <= rom_pan_r[idx];
            v_damper  [load_voice] <= rom_damper[idx];
            v_env     [load_voice] <= '0;
            v_gain    [load_voice] <= GAIN_ONE;
            v_age     [load_voice] <= '0;
            v_level   [load_voice] <= '0;
            pending   <= 1'b0;
            state     <= ST_WAIT;
          end else begin
            load_idx <= load_idx + 1'b1;
          end
        end

        // -- hold out the rest of the sample period ----------------------
        ST_WAIT: begin
          if (period_ctr >= 32'(SAMPLE_PERIOD-1)) begin
            period_ctr   <= '0;
            state        <= ST_STEP;

            // Read this sample out of the arrival buffer and clear the slot so
            // it is empty when the write pointer wraps back onto it.
            dry_l        <= delay_l[delay_wp];
            dry_r        <= delay_r[delay_wp];
            delay_l[delay_wp] <= '0;
            delay_r[delay_wp] <= '0;
            delay_wp     <= delay_wp + 1'b1;
            sample_valid <= 1'b1;

            // Envelopes advance once per audio sample.
            for (int v = 0; v < NUM_VOICES; v++) begin
              if (v_active[v]) begin
                automatic logic signed [63:0] rise, fall;
                rise     = ((64'($signed(GAIN_ONE)) - 64'($signed(v_env[v])))
                            * 64'($signed(v_attack[v]))) >>> GAIN_Q;
                v_env[v] <= v_env[v] + GAIN_W'(rise);
                v_age[v] <= v_age[v] + 1'b1;
                if (v_released[v]) begin
                  fall      = (64'($signed(v_gain[v]))
                               * 64'($signed(v_damper[v]))) >>> GAIN_Q;
                  v_gain[v] <= GAIN_W'(fall);
                  if (fall < 64'sd64) v_active[v] <= 1'b0;
                end
              end
            end
          end
        end

        default: state <= ST_STEP;
      endcase
    end
  end

endmodule

`default_nettype wire
