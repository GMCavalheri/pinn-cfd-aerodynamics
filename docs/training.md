# Training and the compute-budget record

Code: [`src/pinncfd/training.py`](../src/pinncfd/training.py),
[`src/pinncfd/nn.py`](../src/pinncfd/nn.py),
[`src/pinncfd/autodiff.py`](../src/pinncfd/autodiff.py),
[`src/pinncfd/io.py`](../src/pinncfd/io.py) ·
Column guide: [`validation/results/README.md`](../validation/results/README.md)

Every PINN in this project is trained by the same loop, and every run writes one row to
`validation/results/compute_budget.csv`. This note defines what those rows measure.

## The network

[`MLP`](../src/pinncfd/nn.py) maps inputs through an affine scaling $(x - c)/s$, an optional
Fourier-feature layer $[\sin 2\pi xB, \cos 2\pi xB]$ with a fixed Gaussian $B$ (Tancik et al.
2020), `depth` hidden layers of `width` units (tanh, sin or SiLU), and a linear output.
Weights use Glorot-normal initialisation; sine networks use SIREN's hidden-layer rule with
$\omega_0 = 1$ throughout (Sitzmann et al. 2020), since these fields are smooth.
Derivatives of the output with respect to the inputs come from `torch.autograd.grad` with
`create_graph=True`, so they can enter a loss and be differentiated again
([`autodiff.py`](../src/pinncfd/autodiff.py): gradient, Laplacian, partials).

## The loss

A problem provides named loss terms $\mathcal L_k$, each a mean square over its collocation
points, and weights $\lambda_k$:

```math
\mathcal L(\theta) = \sum_k \lambda_k \mathcal L_k(\theta).
```

Collocation points are redrawn every `resample_every` Adam steps, so the network cannot overfit
a fixed point set.

## Two optimiser stages

1. **Adam** (Kingma & Ba 2015) for `adam_steps`, with the learning rate decaying exponentially
   from `lr` to `lr_final`. Adam is robust far from a minimum but stalls near one.
2. **L-BFGS** (Liu & Nocedal 1989) for up to `lbfgs_iters` iterations, with a strong-Wolfe
   line search and history 50, on the last batch of points. It runs in chunks of `eval_every`
   iterations so the error can be logged. `max_eval` is set to ten times the chunk, because
   PyTorch's default (1.25 × `max_iter`) ends a chunk early, which would look like convergence.
   A non-finite loss rolls the network back to the last good state and stops.

Near a minimum L-BFGS converges far faster than Adam: in every phase of this project the error
curve drops sharply when it starts.

## What a run records

The validation error is evaluated every `eval_every` steps, outside the timed region. Steps
are counted globally: Adam steps $1, \dots, A$, then L-BFGS iterations $A + 1, \dots, A + L$.

| Quantity | Definition |
|---|---|
| Error | The phase's headline validation error at the last step |
| Training time | Wall time per stage, CUDA-synchronised, evaluation excluded |
| Steps to target | First evaluated step with error ≤ target (1e-2 by default; also 1e-1, 3e-2 and 1e-3 in `metrics_json`) |
| Plateau step | First evaluated step $s$ from which the running-best error never improves by more than `plateau_tol` (1 %) over any later window of `plateau_window` steps |

If the plateau step equals the last step, the run was still improving when the budget ran out.
"First step below target" counts transient dips: an error that touches the target and then
rises again still counts. The convergence plots show when this happened.

`record_run` writes one row per (phase, case, seed), with the device, library versions, seed,
date and git commit. A rerun replaces its row. All GPU times are for an RTX 2060 clock-capped at
1000 MHz; at this network size a step is limited by kernel-launch latency, not arithmetic, so
time grows with depth more than with width.

## References

- D. P. Kingma & J. Ba, "Adam: a method for stochastic optimization", *ICLR* 2015.
- D. C. Liu & J. Nocedal, "On the limited memory BFGS method for large scale optimization",
  *Math. Program.* 45, 1989.
- M. Tancik et al., "Fourier features let networks learn high frequency functions in low
  dimensional domains", *NeurIPS* 2020.
- V. Sitzmann et al., "Implicit neural representations with periodic activation functions",
  *NeurIPS* 2020.
