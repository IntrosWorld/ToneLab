# Port study (measurements only; nothing ported yet)

Machine: 18 logical cores, PyTorch 2.8 CPU. g++ and cmake are installed (conda); no Rust, no MSVC, no tuned FFT library.
Files: measure.py -> python_measurements.json; flow_bench.cpp -> cpp_measurements.txt.

## Python today (amp resonant flow, full audio, 104 s of test recordings)
frame 46 ms : 4.9x faster than real time (14 threads), 2.9x on one thread; waveform error 0.154 / 0.157 / 0.100
Other models: chorus 25x real time; per-note flow 34 ms per 2.5 s note (75x).
Cost of one flow step for one frame (one thread): 20 inverse FFTs 0.091 ms, 8 products 0.043, 8 forward FFTs 0.029,
damping 0.008 -> 0.17 ms; 8 steps = 1.4 ms per frame = 12 % of one core at 86 frames/s (core loop only).

## Shorter frames with the SAME weights (no retraining)
frame 23 ms : error 0.161 / 0.151 / 0.104  (unchanged)
frame 12 ms : error 0.194 / 0.193 / 0.128
frame  6 ms : error 0.252 / 0.232 / 0.162
frame  3 ms : error 0.332 / 0.428 / 0.357
Latency is about one frame. 12-23 ms is reachable without retraining; under 10 ms needs retraining or a redesign.

## C++ core loop, naive (own radix-2 FFT, random weights, one thread, g++ -O3 -march=native)
frame 46 ms: 11.3 ms per frame = 1.0x real time.  That is 8x SLOWER than PyTorch's core loop (1.4 ms), because PyTorch's
FFT is already tuned C++.  FFTs are ~70 % of the cost, so a C++ port only pays off with a tuned FFT library
(PFFFT, pocketfft, FFTW); expected then: about the PyTorch core-loop figure. Not measured (no such library installed).
The benchmark does not load the trained weights and produces no audio.
