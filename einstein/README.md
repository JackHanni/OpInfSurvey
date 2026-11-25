# OpInf on the Einstein field equations: binary black hole SXS:BBH:0305

We apply Operator Inference to a **published** numerical-relativity solution of the vacuum
Einstein equations: the merger of two black holes, from the SXS catalog. Nothing here solves
Einstein's equations. We load the solution, fit OpInf on one time window, predict forward, and
compare with the true solution.

## Crash course: modeling Einstein's equations

This section is for readers who know numerical PDEs and linear algebra but not general relativity.
It covers just enough to follow the rest of this README.

**Units.** Numerical relativity uses geometric units, $G = c = 1$, in which mass, length and time
are all the same kind of quantity. For a binary, everything is measured in units of the total mass
$M$ of the two black holes:

$$
1\,M \text{ of time} = \frac{GM}{c^3} \approx 4.93\,\mu\text{s} \times \frac{M}{M_\odot},
\qquad
1\,M \text{ of length} = \frac{GM}{c^2} \approx 1.48\,\text{km} \times \frac{M}{M_\odot}.
$$

The vacuum Einstein equations have no built-in length scale, so one simulation describes a binary
of *any* total mass: pick $M$ and the times and lengths follow. For a GW150914-like binary
($M \approx 65\,M_\odot$), $1\,M \approx 0.32$ ms, so the 3500 M of data used here last about a
second and the ringdown window (30 M) about 10 ms. For a $10^6\,M_\odot$ supermassive binary the
same data would span about 5 hours. The strain is likewise given as $r h / M$, the strain at
distance $r$ rescaled, which is why its values are around 0.1 rather than the $10^{-21}$ measured
at Earth.

### 1. What the equations say

The unknown is the **metric** $g_{\mu\nu}(x)$: a symmetric $4\times4$ matrix field (10 functions of space and time)
that defines distances and times, $ds^2 = g_{\mu\nu}\,dx^\mu dx^\nu$. Einstein's equations are

$$
G_{\mu\nu} = 8\pi T_{\mu\nu}, \qquad G_{\mu\nu} = R_{\mu\nu} - \tfrac{1}{2} R\, g_{\mu\nu}.
$$

- $T_{\mu\nu}$ is the matter content (energy density, momentum, stress). In **vacuum** $T_{\mu\nu} = 0$, and
  spacetime can still be curved and dynamic: gravitational waves are vacuum solutions.
- $G_{\mu\nu}$ (the Einstein tensor) is built from first and second derivatives of $g_{\mu\nu}$. The equations are
  a **nonlinear second-order PDE system** for $g$. They are linear in the second derivatives
  (quasilinear), but the coefficients depend on $g$ and its inverse.
- Analogy: in Burgers', $u_t + u\,u_x = \nu\,u_{xx}$, the unknown field also sets the coefficients of its
  own equation. Einstein's equations do the same, with the inverse metric $g^{\mu\nu}$ playing the role
  of $u$ in $u\,u_x$. The inverse metric is where the non-polynomial terms come from.

### 2. Why it is hard numerically

1. **Coordinate (gauge) freedom.** Any change of coordinates gives another valid $g_{\mu\nu}$ for the same
   physical spacetime. Four of the ten functions are pure coordinate choices, so the equations alone
   don't determine a unique solution. You must pick a gauge, and a bad one can make a smooth
   physical solution look singular on the grid.
2. **Constraints and evolution.** Split spacetime into slices of constant time (the **3+1** or ADM
   split):

$$
ds^2 = -\alpha^2\,dt^2 + \gamma_{ij}\,(dx^i + \beta^i dt)(dx^j + \beta^j dt)
$$

   Here $\gamma_{ij}$ is the metric within a slice, $\alpha$ (lapse) sets how far apart in proper time the slices
   are, and $\beta^i$ (shift) sets how spatial coordinates move between slices. Einstein's equations then
   divide into:

   - **evolution equations** (hyperbolic-like) for $\gamma_{ij}$ and its time derivative, the extrinsic
     curvature $K_{ij}$; and
   - **constraint equations** (elliptic) that every slice must satisfy, for example the Hamiltonian
     constraint $R + K^2 - K_{ij}K^{ij} = 16\pi\rho$.

   The exact evolution preserves the constraints, but discretization error does not. Constraint
   violation can grow and wreck a simulation, so monitoring it is a standard accuracy check.
   
3. **Well-posedness.** The original ADM evolution equations are only weakly hyperbolic and tend to
   blow up numerically. Production codes use reformulations (BSSN, generalized harmonic) that are
   strongly hyperbolic. This is part of why full 3D numerical relativity took decades.

4. **Singularities.** Black holes and big-bang/big-crunch singularities are places where curvature
   blows up. Codes must avoid them (excision, special gauges) or approach them carefully.

### 3. Binary black holes and what the catalog publishes

- **The simulation.** Two black holes on a quasi-circular orbit are evolved with the full 3+1
  equations: generalized-harmonic formulation, spectral methods, black-hole interiors excised.
  The SXS collaboration does this with the Spectral Einstein Code (SpEC). One run costs weeks on a
  cluster, which is why we load results instead of producing them.
- **Three phases.**
  1. **Inspiral:** the orbit shrinks as gravitational waves carry energy away, and the wave frequency
     and amplitude rise. This is the *chirp*.
  2. **Merger:** a common horizon forms. The dynamics here are strongly nonlinear.
  3. **Ringdown:** the remnant is a single, perturbed **Kerr** (rotating) black hole that settles
     down by emitting damped oscillations, its **quasinormal modes (QNMs)**. Perturbation theory says
     the ringdown is a sum of terms $A\,e^{(-\omega_I + i\omega_R)t}$, whose frequencies depend only on the
     remnant's mass and spin. So the ringdown is **linear dynamics**.
- **What is published: the waveform.** The full spacetime metric on the 3D grid would be terabytes,
  so catalogs publish the gravitational-wave *strain* far from the source,
  $h = h_+ - i\,h_\times$, as a function of time and direction on the sky $(\theta, \phi)$. It is expanded in
  spin-weighted spherical harmonics,

$$
h(t, \theta, \phi) = \sum_{\ell \ge 2} \sum_{|m| \le \ell} h_{\ell m}(t)\, {}_{-2}Y_{\ell m}(\theta, \phi).
$$

  The mode amplitudes $h_{\ell m}(t)$ are the data. They are a time-dependent field on the sphere,
  discretized spectrally, which is exactly the kind of state OpInf works with.
- **Gravitational-wave memory.** Besides oscillating, the $m = 0$ modes pick up a *permanent* offset
  during merger. This is a real nonlinear GR effect, but it is non-oscillating and dwarfs the
  decaying ringdown in any norm. See "The data" below for how it is handled.

### 4. How a published solution is trusted

1. **Several resolutions** (Lev0 … Lev6 for this simulation). Differences between levels estimate
   the numerical error. We use the finest, Lev6.
2. **Constraint monitoring:** the constraint equations (Section 2) must stay small and converge to
   zero with resolution.
3. **Extrapolation to infinity:** the waveform is extracted on spheres at finite radius and
   extrapolated in $1/r$. "N2" means a 2nd-order polynomial in $1/r$, the catalog default.
4. **Physical cross-checks.** The remnant's ringdown frequency must match Kerr perturbation theory
   for the measured final mass and spin. We repeat this check *with OpInf* below.

### 5. Further reading

- S. Carroll, *Spacetime and Geometry* (Addison-Wesley, 2004): general relativity from scratch.
- M. Alcubierre, *Introduction to 3+1 Numerical Relativity* (Oxford University Press, 2008).
- T. Baumgarte & S. Shapiro, *Numerical Relativity: Solving Einstein's Equations on the Computer*
  (Cambridge University Press, 2010).
- E. Gourgoulhon, *3+1 Formalism in General Relativity* (Springer Lecture Notes in Physics 846,
  2012); free version at arXiv:gr-qc/0703035.
- M. Boyle et al., *The SXS Collaboration catalog of binary black hole simulations*,
  Class. Quantum Grav. 36, 195006 (2019), arXiv:1904.04831: the catalog and its accuracy checks.
- E. Berti, V. Cardoso & C. Will, *On gravitational-wave spectroscopy of massive black holes with
  the space interferometer LISA*, Phys. Rev. D 73, 064030 (2006): the QNM fits used below.

---

Nothing here solves Einstein's equations. The data is a published numerical-relativity solution
from the **SXS catalog**. We load it, fit OpInf on one time window, predict forward, and compare
with the true solution.

## The data

- **Simulation:** SXS:BBH:0305 (SXS Collaboration, Spectral Einstein Code), highest resolution Lev6.
  Two spinning black holes (mass ratio 1.22, aligned spins 0.33 and −0.44) inspiral and merge.
  It is the GW150914-like configuration. Record: [https://doi.org/10.26138/SXS:BBH:0305](https://doi.org/10.26138/SXS:BBH:0305) (CaltechDATA).
- **What is published:** the gravitational-wave strain at infinity,
  $h(t, \theta, \phi) = \sum h_{\ell m}(t)\, {}_{-2}Y_{\ell m}(\theta, \phi)$, as 77 complex modes ($\ell = 2, \dots, 8$), extrapolated to infinite
  radius ($N = 2$). It is a time-dependent field on the sphere, in units of the total mass $M$.
- **State used for OpInf:** $x(t) = [\mathrm{Re} h_{\ell m};\ \mathrm{Im} h_{\ell m}]$ for the 70 modes with $m \ne 0$, so $x \in \mathbb{R}^{140}$.
  The harmonics are orthonormal, so $\|x\|$ is the $L^2$ norm of $h$ on the sphere.
- **$m = 0$ modes left out:** after merger they are dominated by the gravitational-wave *memory*, a
  permanent offset ($h_{20} \approx 0.08$), while the oscillating part decays to around 1e-3. Kept in, the
  offset dominates every error norm and the POD basis; on a first run it made the ringdown error
  look like 100 %. They are also the least reliable modes in extrapolated SXS waveforms.
  `load_sxs(drop_m0 = false)` keeps them.

## How to run

```bash
pip install sxs h5py                   # only for decoding SXS's compressed file format
python einstein/fetch_sxs.py   # download + export to data/sxs_bbh_0305.h5 (≈ 18 MB, gitignored)
julia  einstein/opinf_sxs.jl   # OpInf; writes output/sxs_log.txt and figures
```

Python is used **only** to decode the download. SXS stores waveforms in a compressed
rotating-frame format that only the official `sxs` package reads. The export is plain HDF5, so it
loads in Julia with JLD2 (transposed, as usual) and in Python with h5py.
`fetch_sxs.py` resamples to uniform $\Delta t = 0.5M$, drops $t < 300M$ (the initial "junk radiation"), and
also exports $\dot h$, the derivative of the spline through the published data. So OpInf uses no
finite-difference derivatives.

## Method

- POD basis from the training window. The model is linear, $\dot{\hat x} = \hat A\hat x$, or quadratic,
  $\dot{\hat x} = \hat A\hat x + \hat H(\hat x \otimes \hat x)$. The fit is Tikhonov least squares by Householder QR of $\begin{bmatrix} D \\ \sqrt{\lambda}\, I \end{bmatrix}$.
- $\lambda$ is chosen **from training data only**: fit on the first 75 % of the window and pick the λ whose
  ROM best forecasts the last 25 %. Then refit on the whole window.
- The ROM is integrated with RK4 from the first training snapshot, through the training window
  and into the prediction window. Errors are relative 2-norm errors over all 140 components.
  "POD floor" is the error of projecting the true prediction-window data onto the training basis.

## Results (`output/sxs_log.txt`)

Merger (peak of $|h_{22}|$) is at $t_{\text{peak}} = 3692.5M$.

### 1. Ringdown: train $t_{\text{peak}} + [20, 50]$, predict $t_{\text{peak}} + (50, 80]$

| r  | linear: train / predict | quadratic: train / predict | POD floor (predict) |
| -- | ----------------------- | -------------------------- | ------------------- |
| 2  | 8.7e-2 / 3.3e-1         | 8.3e-2 / 2.5e-1            | 1.4e-1              |
| 8  | 1.5e-2 / 1.7e-1         | 1.3e-2 / 9.9e-2            | 8.0e-2              |
| 16 | 3.7e-3 / 1.3e-1         | —                         | 1.6e-2              |
| 20 | 1.5e-3 / 1.0e-1         | —                         | 1.1e-2              |

Figures: `ringdown_h22.png` (truth vs ROMs, plus error vs t) and `ringdown_error_vs_r.png`.
Both ROMs track the $h_{22}$ waveform closely through the whole window. For the linear model, pointwise
error is about 1 % at the end of training, then grows to 10–40 % as the signal decays by another
factor of 10.

**Physics check (`ringdown_qnm.png`).** After merger the remnant is a perturbed Kerr black hole, and
perturbation theory says the ringdown is a sum of damped exponentials (quasinormal modes). That
makes it linear dynamics, so the learned $\hat A$ should contain the QNM frequency. For the remnant
($M_f = 0.952$, $\chi_f = 0.692$ from the SXS metadata), the fit of Berti, Cardoso & Will (2006) gives
$\omega_{220} = -0.0858 \pm 0.5584i$. The closest eigenvalue of the learned $\hat A$:

| training starts at          | $r = 2$  | $r = 16$ |
| --------------------------- | -------- | -------- |
| $t_{\text{peak}} + 0$       | 22 % off | 15 % off |
| $t_{\text{peak}} + 10$      | 9.3 %    | 5.8 %    |
| $t_{\text{peak}} + 20$      | 2.5 %    | 2.0 %    |
| $t_{\text{peak}} + 30$      | 0.8 %    | 0.8 %    |

OpInf recovers the black hole's ringing frequency from the data alone, and the agreement improves
steadily as the training window moves later. That is exactly what you'd expect: right after merger
the dynamics are still nonlinear, and they become linear QNM ringing over roughly 20–30 M. This is
a clear case of "the model form has to match the physics".

### 2. Inspiral: train [3300, 3500] M, predict (3500, 3600] M

| r | linear: train / predict | quadratic: train / predict | POD floor (predict) |
| - | ----------------------- | -------------------------- | ------------------- |
| 2 | 4.2e-1 / 9.7e-1         | 4.1e-1 / 9.8e-1            | 4.2e-2              |
| 6 | 4.2e-1 / 9.4e-1         | **7.7e-3 / 3.9e-1**  | 1.1e-2              |
| 8 | 4.2e-1 / 9.4e-1         | 1.9e-3 / blows up          | 3.4e-3              |

Figure: `inspiral_h22.png`.

- The **linear model can't even fit the training window**. A linear system has fixed frequencies,
  and the inspiral chirps: the frequency and amplitude rise as the black holes spiral in. Here $\lambda$
  sits at the top of the grid ($10^{-2}$) because no $\lambda$ fixes this.
- The **quadratic model at $r = 6$** fits training to 0.8 % and predicts the first ~$40M$
  (about half a cycle of $h_{22}$) to within 5 %. It then falls behind the accelerating chirp, reaching 39 %
  over the window. With $r \ge 8$ it overfits ($\kappa(D) \ge 4\times10^{10}$) and blows up.
- Interpretation: in terms of $h$, the chirp is not a quadratic system. To leading order the
  frequency grows like $\omega \propto |h|^{3/2}$, which is non-polynomial. A lifting to new variables would be needed,
  e.g. ones that include the amplitude and frequency.

## Takeaways

1. OpInf works on a real, published Einstein-equation solution when the physics behind the data
   matches the model form. The ringdown is linear, and the learned $\hat A$ recovers the Kerr QNM
   frequency to within 1–2 %.
2. When the model form is wrong, prediction fails whatever the regression does. A linear model
   can't chirp, and the quadratic form only buys a short horizon.
3. The same conditioning story as Burgers': one trajectory and a quadratic model give $\kappa(D)$ of
   $10^{8}$– $10^{12}$ already at $r = 6$– $8$.

## Caveats

- One simulation, one resolution (Lev6). Lev5 vs Lev6 differences would give an error bar on the
  "true" solution, but that hasn't been done yet.
- The Kerr QNM value comes from a published fit (not an exact Kerr computation) plus the catalog's remnant mass and spin. It isn't an independent computation.
- Training windows were picked by hand (ringdown start $t_{\text{peak}} + 20M$, inspiral $[3300, 3500]$),
  and some alternatives were tried first. The start-time sweep shows how sensitive the ringdown
  result is to that choice.
