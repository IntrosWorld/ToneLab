// Speed test of the resonant-flow core loop in plain C++ (single thread, random weights, same shapes as the trained model:
// 8 steps, 4 pair products, 4 triple products, damping rank 3).  Uses a simple radix-2 FFT written here, so a tuned FFT
// library would be faster.  This measures speed only; it does not load the trained weights or produce audio.
#include <complex>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <random>
using cf = std::complex<float>; using vec = std::vector<cf>;
struct FFT { int n; std::vector<int> rev; vec tw;
  FFT(int n_) : n(n_), rev(n_), tw(n_ / 2) { int lg = 0; while ((1 << lg) < n) lg++; for (int i = 0; i < n; i++) { int r = 0; for (int b = 0; b < lg; b++) if (i >> b & 1) r |= 1 << (lg - 1 - b); rev[i] = r; }
    for (int i = 0; i < n / 2; i++) tw[i] = std::polar(1.0f, (float)(-2 * 3.14159265358979323846 * i / n)); }
  void run(cf* a, bool inv) const { for (int i = 0; i < n; i++) if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    for (int len = 2; len <= n; len <<= 1) { int half = len / 2, step = n / len; for (int i = 0; i < n; i += len) for (int j = 0; j < half; j++) { cf w = tw[j * step]; if (inv) w = std::conj(w); cf u = a[i + j], v = a[i + j + half] * w; a[i + j] = u + v; a[i + j + half] = u - v; } } } };
int main() { const int STEPS = 8, R2 = 4, R3 = 4, NF = 2 * R2 + 3 * R3, RANK = 3; std::mt19937 g(1); std::normal_distribution<float> nd(0, 1);
  for (int N : {4096, 2048, 1024, 512}) { int B = N / 2 + 1, frames = 200; FFT fft(N);
    std::vector<vec> a(NF, vec(B)), q(R2 + R3, vec(B)); vec lin(B), Z(B), T(B); std::vector<std::vector<float>> ka(RANK, std::vector<float>(B)), kb(ka), kp(ka);
    for (auto& v : a) for (auto& x : v) x = cf(nd(g), nd(g)) * 0.1f; for (auto& v : q) for (auto& x : v) x = cf(nd(g), nd(g)) * 0.01f; for (auto& x : lin) x = cf(nd(g), nd(g)) * 0.01f;
    for (int r = 0; r < RANK; r++) for (int k = 0; k < B; k++) { ka[r][k] = std::fabs(nd(g)) * 0.1f; kb[r][k] = std::fabs(nd(g)) * 0.1f; kp[r][k] = nd(g) * 0.1f; }
    std::vector<vec> w(NF, vec(N)), pr(R2 + R3, vec(N)); std::vector<float> p(B); double sink = 0; auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < frames; f++) { for (int k = 0; k < B; k++) Z[k] = cf(nd(g), nd(g)) * 0.01f;
      for (int s = 0; s < STEPS; s++) { float h = 0.25f;
        for (int c = 0; c < NF; c++) { cf* x = w[c].data(); for (int k = 0; k < B; k++) x[k] = Z[k] * a[c][k]; for (int k = 1; k < N / 2; k++) x[N - k] = std::conj(x[k]); fft.run(x, true); }
        for (int c = 0; c < R2; c++) for (int i = 0; i < N; i++) pr[c][i] = cf(w[c][i].real() * w[R2 + c][i].real(), 0);
        for (int c = 0; c < R3; c++) for (int i = 0; i < N; i++) pr[R2 + c][i] = cf(w[2 * R2 + c][i].real() * w[2 * R2 + R3 + c][i].real() * w[2 * R2 + 2 * R3 + c][i].real(), 0);
        for (int k = 0; k < B; k++) T[k] = 0; for (int c = 0; c < R2 + R3; c++) { fft.run(pr[c].data(), false); for (int k = 0; k < B; k++) T[k] += q[c][k] * pr[c][k]; }
        float sk[RANK], sp[RANK]; for (int k = 0; k < B; k++) p[k] = std::norm(Z[k]); for (int r = 0; r < RANK; r++) { float acc = 0; for (int k = 0; k < B; k++) acc += kb[r][k] * p[k]; sk[r] = acc; sp[r] = acc; }
        for (int k = 0; k < B; k++) { float e = 0, ph = 0; for (int r = 0; r < RANK; r++) { e += ka[r][k] * sk[r]; ph += kp[r][k] * sp[r]; } Z[k] = (Z[k] * std::exp(h * lin[k]) * std::polar(1.0f, h * std::tanh(ph)) + h * T[k] / (float)N) / (1 + h * e); } }
      sink += std::abs(Z[5]); }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / frames; double need = 88200.0 / (N / 4);
    std::printf("N %d  frame %.1f ms  hop %.2f ms  compute %.3f ms/frame  frames needed/s %.0f  core load %.1f %%  times faster than real time %.1f  (%g)\n", N, N / 88.2, N / 4 / 88.2, ms, need, ms * need / 10, 1000 / (ms * need), sink); } }
