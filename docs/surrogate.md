# The PINN as a surrogate

Code: [`src/pinncfd/surrogate.py`](../src/pinncfd/surrogate.py),
[`validation/surrogate_benchmark.py`](../validation/surrogate_benchmark.py) ·
Lesson: [Lesson 4](lessons/04-surrogates-in-a-design-loop.md) ·
Notebook: [`04_surrogate_benchmark.ipynb`](../notebooks/04_surrogate_benchmark.ipynb) ·
Results: [`validation/results/surrogate_benchmark.md`](../validation/results/surrogate_benchmark.md)

## Batched lift

- `cl_pressure(model, params, device, n=120)`: Cp from the predicted velocity at the control
  points of an $n$-panel contour of every section, integrated as $-\sum C_p\mathbf n\Delta s$
  and rotated to the wind. This is the panel method's own pressure lift; a test reproduces
  AERO-01's to $10^{-6}$ when fed its surface velocities.
- `cl_circulation(model, params, device, n=256)`: $c_l = 2\Gamma$ from the circulation on a
  circle of radius 1.5, by the periodic trapezoidal rule. No contour is needed.
- `solve_alpha(model, params, target, device)`: Newton's method on $\alpha$ for every row at
  once. $\partial c_l/\partial\alpha$ is the derivative of the pressure lift with respect to the
  network's $\alpha$ input, from autograd. Tested against Brent's method.

## Cost model

With training time $T_{\text{train}}$ and per-case times $t_{\text{PINN}} < t_{\text{solver}}$,
the surrogate is cheaper beyond

```math
n^{*} = \frac{T_{\text{train}}}{t_{\text{solver}} - t_{\text{PINN}}}
```

evaluations. The benchmark reports $n^{*}$ against two references: the 200-panel baseline,
and the coarsest panel method at least as accurate as the PINN.

## Timing

`timed` runs one warm-up call, synchronises CUDA before and after each run, and returns the
median of up to three runs. The PINN timings include building the contours, which take 85 % of
the pressure-lift time at $n = 10^4$.

## Results (Phase 3 champion, RTX 2060 at 1000 MHz)

| | Mean abs cl error | Time per case (n = 10⁴) | Break-even |
|---|---|---|---|
| Panel method, 200 panels | 0.00054 | 8.0 ms | — |
| Panel method, 16 panels | 0.021 | 0.39 ms | — |
| PINN, pressure, GPU | 0.031 | 154 µs | 84,000 vs 200 panels; 2.8 million vs 16 |
| PINN, circulation, GPU | 0.035 | 66 µs | 2.0 million vs 16 panels |
| PINN, pressure, CPU | 0.031 | 278 µs | 85,000 vs 200 panels |

Break-even counts charge the 657 s of training. Charging the 40-minute architecture search
too raises the 200-panel break-even to 391,000. Inverse problem, $c_l = 0.8$ for six sections:
Newton on the PINN, 4 batched evaluations, 0.064 s; Brent on 200 panels, 36 solves, 0.27 s.
The PINN's incidences are 0.05–0.41° off.
