# Computational Physics Project: Signal-Processing Problem Statements (S1-S4)

Project: group of 3, submission Nov 3. Required: (1) pick a problem needing numerics, (2) explain why computation is necessary, (3) implement a method, (4) present and analyze results, (5) at least one extension or parameter study, (6) presentation.

Effort figures are my rough estimates in person-hours for one working implementation plus one extension.

| | Problem | Novelty | Risk | Effort (est.) |
|---|---|---|---|---|
| S1 | Matched-filter detection of a gravitational-wave chirp | medium-high | medium | 25 h |
| S2 | Sparse signal recovery below Nyquist (compressed sensing) | medium | low | 15-20 h |
| S3 | Deconvolution of an instrument response | low-medium | low | 15-20 h |
| S4 | Spectral and nonlinear analysis of a driven damped pendulum | low | lowest | 12-15 h |

**General resources**
- Press et al., *Numerical Recipes* (FFT, spectral estimation chapters).
- Oppenheim and Schafer, *Discrete-Time Signal Processing*.
- `scipy.signal` and `numpy.fft` documentation.
- Prerau et al. or any Welch-method tutorial for PSD estimation.

---

## S1. Matched-filter detection of a gravitational-wave chirp in noise

**Physics.** A compact-binary inspiral radiates a signal whose frequency sweeps upward. At leading (Newtonian) order, with `tau = t_c - t`:

```
f(tau) = (1/pi) * (5 / (256 tau))^(3/8) * (G M_c / c^3)^(-5/8),     M_c = (m1 m2)^(3/5) / (m1 + m2)^(1/5)
h(t) ~ M_c^(5/3) f^(2/3) / D * cos(phase)
```

So `f ~ tau^(-3/8)` and the amplitude grows as `f^(2/3)`. The strain is about 1e-21, far below the noise at every time sample.

**Why computation is necessary.** The noise is not white: it is dominated by seismic noise at low frequency, shot noise at high frequency, and narrow lines. The optimal linear detection statistic is the matched filter. In the frequency domain, with data s~(f), template h~(f) and noise PSD S_n(f):

```
rho(t) = 4 Re integral  s~(f) h~*(f) / S_n(f)  e^{2 pi i f t} df  / sigma
```

evaluated for all time shifts at once by an inverse FFT. There is no analytic shortcut for real data.

**Pipeline**
1. Estimate the PSD with Welch's method (`scipy.signal.welch`).
2. Generate Newtonian-order templates for a chosen chirp mass.
3. Whiten data and template; compute SNR(t) by FFT cross-correlation; locate the peak.
4. Test on injected signals in synthetic colored Gaussian noise first, with an aLIGO-like PSD. Then analyze real GW150914 strain data from GWOSC.

**Extensions and parameter studies**
- Template bank over chirp mass; plot SNR versus template mass mismatch.
- ROC curves (detection probability versus false-alarm rate) versus injected SNR.
- Sensitivity to Welch segment length and window.
- Glitch handling or gating on real data.

**Honest caveats**
- Newtonian-order templates recover only part of the true SNR for a real event like GW150914, because the late inspiral, merger and ringdown matter. State this openly; do not claim a perfect recovery.
- Real data needs a download (internet access); `gwpy` makes this easy. Have the data cached before presentation day.

**Resources**
- GWOSC: gwosc.org and gwosc.org/tutorials, including the GW150914 signal-processing tutorial notebook.
- PyCBC tutorials.
- Allen et al., "FINDCHIRP", Phys. Rev. D 85, 122006 (2012).
- Maggiore, *Gravitational Waves, Vol. 1* (chapter 4).

---

## S2. Sparse signal recovery below the Nyquist rate (compressed sensing)

**Physics.** A spectrum with only a few lines (NMR, spectroscopy, interferometry) is sampled at randomly chosen times, at a rate far below the Nyquist rate. Classical sampling theory says the full spectrum cannot be recovered; sparsity says it can.

**Why computation.** The recovery problem is an underdetermined linear system with infinitely many solutions. Picking the sparsest one is combinatorial, and its convex relaxation (L1 minimization) has no closed-form solution. It requires an iterative optimization algorithm.

**Method.** Solve

```
min_x  1/2 || A x - y ||_2^2 + lambda || x ||_1
```

where A is a partial (non-uniform) Fourier matrix built from the random sample times.
- Implement **ISTA** yourself: `x <- soft_threshold( x - t * A^H (A x - y), t * lambda )`.
- Upgrade to **FISTA** (adds Nesterov momentum) and compare convergence rates.
- Baselines: naive FFT of the irregular samples (heavy aliasing), and Lomb-Scargle periodogram.

**Extensions and parameter studies**
- Phase-transition plot: probability of exact recovery versus sparsity ratio k/M and undersampling ratio M/N.
- Noise robustness: reconstruction error versus SNR.
- Choice of lambda (cross-validation or L-curve).
- Debiasing step: re-fit amplitudes on the recovered support.

**Assessment.** The conceptually cleanest signal-processing option, and implementable in under 50 lines. The physics content is thinner than S1, so motivate it with a concrete experiment (for example NMR with non-uniform sampling).

**Resources**
- Candes and Wakin, "An Introduction to Compressive Sampling", IEEE Signal Processing Magazine (2008).
- Donoho, "Compressed Sensing", IEEE Trans. Inf. Theory (2006).
- Beck and Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm" (FISTA), SIAM J. Imaging Sci. (2009).
- Scargle (1982) or the `astropy.timeseries.LombScargle` docs for the baseline.

---

## S3. Deconvolution of an instrument response (Richardson-Lucy vs Wiener vs Tikhonov)

**Problem.** A measured signal is `y = h * x + noise`, where `h` is the instrument blur (spectrometer slit function, telescope PSF, detector response). Naive inversion in Fourier space, `X = Y / H`, amplifies noise catastrophically wherever `H` is small. This is the textbook ill-posed inverse problem.

**Why computation.** No closed-form stable inverse exists. Every practical method trades resolution against noise amplification, and the trade-off must be tuned and measured numerically.

**Methods to compare**
- **Wiener filter** (frequency domain, needs a noise-to-signal estimate).
- **Tikhonov regularization** with the parameter chosen from the **L-curve**.
- **Richardson-Lucy** (iterative, positivity preserving; the iteration count acts as the regularizer).

**Test signals.** Synthetic spectrum with overlapping Gaussian/Lorentzian peaks and a known blur kernel, so you know the ground truth.

**Metrics.** MSE, peak-position error, ringing amplitude, noise amplification as a function of SNR.

**Extensions and parameter studies**
- Error versus SNR for each method (a clean parameter study).
- L-curve and generalized cross-validation for choosing lambda.
- Effect of kernel mis-specification (deconvolve with the wrong width).
- 2D version on an image.

**Assessment.** Easy to implement but easy to leave superficial. The L-curve analysis, the noise study, and the kernel-mismatch test carry the depth.

**Resources**
- Hansen, *Discrete Inverse Problems: Insight and Algorithms*.
- Richardson, J. Opt. Soc. Am. 62, 55 (1972); Lucy, Astron. J. 79, 745 (1974).
- `skimage.restoration` (`wiener`, `richardson_lucy`) as reference implementations. Write your own first, then compare.

---

## S4. Spectral and nonlinear analysis of a driven damped pendulum

**Physics.** The driven damped pendulum

```
theta'' + theta' / q + sin(theta) = g cos(omega_D t)
```

is the standard example of a deterministic system with regular and chaotic regimes. A common textbook choice is `q = 2`, `omega_D = 2/3`, with the drive strength g swept from about 0.9 to 1.5 (regular to chaotic behaviour); verify the regimes numerically.

**Why computation.** The nonlinear term makes the equation analytically unsolvable, and chaotic trajectories can only be characterized statistically from numerical solutions.

**Method**
- Integrate with RK4 (and compare against `scipy.integrate.solve_ivp`).
- Power spectra of theta(t) (FFT, Welch) in regular versus chaotic regimes.
- Poincare sections sampled at the drive period.
- Bifurcation diagram versus g.
- Largest Lyapunov exponent (Wolf et al. algorithm or tangent-space method).
- Short-time Fourier transform or wavelets for transient behaviour.

**Extensions and parameter studies**
- Lyapunov exponent versus g (sign change marks the onset of chaos).
- Sensitivity to initial conditions (divergence of two nearby trajectories).
- Convergence of RK4 with time step; energy-drift check in the undamped limit.

**Assessment.** The easiest and lowest-risk option, but the most standard textbook project and the weakest on signal processing; spectral analysis is a tool here rather than the main subject.

**Resources**
- Giordano and Nakanishi, *Computational Physics* (chaos chapter).
- Strogatz, *Nonlinear Dynamics and Chaos*.
- Wolf, Swift, Swinney, Vastano, "Determining Lyapunov exponents from a time series", Physica D 16, 285 (1985).

---

## Recommendation

- **S1** is the best signal-processing option: the physics and the computation are tightly coupled, the extensions are natural, and it has a real-data payoff.
- **S2** is the safer choice if you want a short implementation and rich parameter studies.
- S3 and S4 are low-risk fallbacks; S4 is the weakest on signal processing.
- Whichever you pick, state the limitations explicitly (template accuracy for S1, noise and mismatch sensitivity for S3, and so on). The rubric explicitly rewards evaluating the strengths and limits of the method.