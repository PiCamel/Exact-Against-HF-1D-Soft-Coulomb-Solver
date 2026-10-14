# Computational Physics Project: Signal-Processing Problem Statements (S1–S4)

Project: group of 3, submission Nov 3. Required: (1) pick a problem needing numerics, (2) explain why computation is necessary, (3) implement a method, (4) present and analyze results, (5) at least one extension or parameter study, (6) presentation.

Effort figures are my rough estimates in person-hours for one working implementation plus one extension.

| | Problem | Novelty | Risk | Effort (est.) |
|---|---|---|---|---|
| S1 | Matched-filter detection of a gravitational-wave chirp | medium–high | medium | 25 h |
| S2 | Sparse signal recovery below Nyquist (compressed sensing) | medium | low | 15–20 h |
| S3 | Deconvolution of an instrument response | low–medium | low | 15–20 h |
| S4 | Spectral and nonlinear analysis of a driven damped pendulum | low | lowest | 12–15 h |

**General resources**

- Press et al., *Numerical Recipes* (FFT and spectral-estimation chapters).
- Oppenheim and Schafer, *Discrete-Time Signal Processing*.
- `scipy.signal` and `numpy.fft` documentation.
- Any Welch-method tutorial for PSD estimation.

---

## S1. Matched-filter detection of a gravitational-wave chirp in noise

**Physics.** A compact-binary inspiral radiates a signal whose frequency sweeps upward. At leading (Newtonian) order, with $\tau = t_c - t$ the time to coalescence and chirp mass

$$
\mathcal M_c = \frac{(m_1 m_2)^{3/5}}{(m_1+m_2)^{1/5}},
$$

the gravitational-wave frequency and strain amplitude scale as

$$
f(\tau) = \frac1\pi\left(\frac{5}{256\,\tau}\right)^{3/8}\left(\frac{G\mathcal M_c}{c^3}\right)^{-5/8},
\qquad
h(t) \propto \frac{\mathcal M_c^{5/3}\,f^{2/3}}{D}\cos\Phi(t)
$$

So $f \propto \tau^{-3/8}$ and the amplitude grows as $f^{2/3}$. The strain is about $10^{-21}$, far below the noise at every time sample.

**Why computation is necessary.** The noise is not white: it is dominated by seismic noise at low frequency, shot noise at high frequency, and narrow lines. The optimal linear detection statistic is the matched filter. With data $\tilde s(f)$, template $\tilde h(f)$ and one-sided noise PSD $S_n(f)$:

$$
\rho(t) = \frac{4}{\sigma}\,\mathrm{Re}\int_0^\infty \frac{\tilde s(f)\,\tilde h^*(f)}{S_n(f)}\,e^{2\pi i f t}\,df,
\qquad
\sigma^2 = 4\int_0^\infty \frac{|\tilde h(f)|^2}{S_n(f)}\,df
$$

This is evaluated for all time shifts at once by an inverse FFT. There is no analytic shortcut for real data.

**Pipeline**

1. Estimate the PSD with Welch's method (`scipy.signal.welch`).
2. Generate Newtonian-order templates for a chosen chirp mass.
3. Whiten data and template; compute $\rho(t)$ by FFT cross-correlation; locate the peak.
4. Test on injected signals in synthetic colored Gaussian noise first, with an aLIGO-like PSD. Then analyze real GW150914 strain data from GWOSC.

**Extensions and parameter studies**

- Template bank over chirp mass; plot peak SNR versus template mass mismatch.
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

**Why computation.** The recovery problem is an underdetermined linear system $\mathbf y = \mathbf A\mathbf x$ with $M < N$ equations, hence infinitely many solutions. Picking the sparsest one ($\min\|\mathbf x\|_0$) is combinatorial, and its convex relaxation (L1 minimization) has no closed-form solution. It requires an iterative optimization algorithm. A typical recovery condition for a $k$-sparse signal is

$$
M \gtrsim C\,k\,\log(N/k)
$$

**Method.** Solve the LASSO problem

$$
\min_{\mathbf x}\ \frac12\|\mathbf A\mathbf x - \mathbf y\|_2^2 + \lambda\|\mathbf x\|_1
$$

where $\mathbf A$ is a partial (non-uniform) Fourier matrix built from the random sample times.

- Implement **ISTA** yourself, with step size $t \le 1/\|\mathbf A\|_2^2$ and soft-thresholding operator $\mathcal S$:
$$
\mathbf x^{(k+1)} = \mathcal S_{t\lambda}\!\left(\mathbf x^{(k)} - t\,\mathbf A^{H}(\mathbf A\mathbf x^{(k)} - \mathbf y)\right),
\qquad
\mathcal S_\theta(z) = \mathrm{sign}(z)\max(|z|-\theta,\,0)
$$
- Upgrade to **FISTA** (adds Nesterov momentum) and compare convergence rates: $O(1/k)$ for ISTA versus $O(1/k^2)$ for FISTA in objective value.
- Baselines: naive FFT of the irregular samples (heavy aliasing), and the Lomb–Scargle periodogram.

**Extensions and parameter studies**

- Phase-transition plot: probability of exact recovery versus sparsity ratio $k/M$ and undersampling ratio $M/N$.
- Noise robustness: reconstruction error versus SNR.
- Choice of $\lambda$ (cross-validation or L-curve).
- Debiasing step: re-fit amplitudes on the recovered support.

**Assessment.** The conceptually cleanest signal-processing option, and implementable in under 50 lines. The physics content is thinner than S1, so motivate it with a concrete experiment (for example NMR with non-uniform sampling).

**Resources**

- Candès and Wakin, "An Introduction to Compressive Sampling", IEEE Signal Processing Magazine (2008).
- Donoho, "Compressed Sensing", IEEE Trans. Inf. Theory (2006).
- Beck and Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm" (FISTA), SIAM J. Imaging Sci. (2009).
- `astropy.timeseries.LombScargle` docs for the baseline.

---

## S3. Deconvolution of an instrument response (Richardson–Lucy vs Wiener vs Tikhonov)

**Problem.** A measured signal is

$$
y(t) = (h * x)(t) + \eta(t)
\quad\Longleftrightarrow\quad
Y(f) = H(f)\,X(f) + N(f)
$$

where $h$ is the instrument blur (spectrometer slit function, telescope PSF, detector response) and $\eta$ is noise. Naive inversion in Fourier space, $\hat X = Y/H = X + N/H$, amplifies noise catastrophically wherever $|H|$ is small. This is the textbook ill-posed inverse problem.

**Why computation.** No closed-form stable inverse exists. Every practical method trades resolution against noise amplification, and the trade-off must be tuned and measured numerically.

**Methods to compare**

- **Wiener filter** (frequency domain; needs signal and noise power spectra $S_x$, $S_\eta$):
$$
\hat X(f) = \frac{H^*(f)}{|H(f)|^2 + S_\eta(f)/S_x(f)}\,Y(f)
$$
- **Tikhonov regularization**, with the parameter $\lambda$ chosen from the **L-curve** (plot $\log\|\mathbf x_\lambda\|$ against $\log\|\mathbf H\mathbf x_\lambda - \mathbf y\|$ and pick the corner):
$$
\hat{\mathbf x}_\lambda = \arg\min_{\mathbf x}\ \|\mathbf H\mathbf x - \mathbf y\|_2^2 + \lambda^2\|\mathbf x\|_2^2
$$
- **Richardson–Lucy** (iterative, positivity preserving; the iteration count acts as the regularizer), with $\check h(t) = h(-t)$:
$$
x^{(k+1)} = x^{(k)}\cdot\left[\check h * \frac{y}{h * x^{(k)}}\right]
$$

**Test signals.** Synthetic spectrum with overlapping Gaussian/Lorentzian peaks and a known blur kernel, so you know the ground truth.

**Metrics.** MSE, peak-position error, ringing amplitude, noise amplification as a function of SNR.

**Extensions and parameter studies**

- Error versus SNR for each method (a clean parameter study).
- L-curve and generalized cross-validation for choosing $\lambda$.
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

$$
\ddot\theta + \frac1q\,\dot\theta + \sin\theta = g\cos(\omega_D t)
$$

is the standard example of a deterministic system with regular and chaotic regimes. A common textbook choice is $q = 2$, $\omega_D = 2/3$, with the drive strength $g$ swept from about 0.9 to 1.5 (regular to chaotic behaviour); verify the regimes numerically.

**Why computation.** The nonlinear term $\sin\theta$ makes the equation analytically unsolvable, and chaotic trajectories can only be characterized statistically from numerical solutions.

**Method**

- Integrate with RK4 (and compare against `scipy.integrate.solve_ivp`). Write the system as first order in $(\theta, \omega, \phi)$ with $\phi = \omega_D t$.
- Power spectra of $\theta(t)$ (FFT, Welch) in regular versus chaotic regimes.
- Poincaré sections sampled at the drive period $T = 2\pi/\omega_D$.
- Bifurcation diagram versus $g$.
- Largest Lyapunov exponent (Wolf et al. algorithm or tangent-space method):
$$
\lambda_{\max} = \lim_{t\to\infty}\frac1t\ln\frac{|\delta(t)|}{|\delta(0)|}
$$
- Short-time Fourier transform or wavelets for transient behaviour.

**Extensions and parameter studies**

- $\lambda_{\max}$ versus $g$ (sign change marks the onset of chaos).
- Sensitivity to initial conditions (divergence of two nearby trajectories).
- Convergence of RK4 with time step $\Delta t$ (error scales as $\Delta t^4$); energy-drift check in the undamped, undriven limit.

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