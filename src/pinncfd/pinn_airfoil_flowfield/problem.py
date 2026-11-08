"""Loss terms, collocation and reference data for the airfoil-family PINN.

Loss terms (each a mean square):

* ``pde``   — Laplace residual of the perturbation psi' at interior points (the known part is
  harmonic away from its vortex, which lies inside the body);
* ``body``  — psi - psi_b on the surface: the body is a streamline;
* ``tangency`` — normal velocity on the surface. The body condition
  psi = psi_b alone charges a leak through the surface by the flux Q it carries: a first
  version without this term let the freestream pass through the thin tail instead of
  developing circulation. Penalising the normal velocity charges the leak speed Q / l, l the
  length of surface crossed, which is large for a leak through a short stretch of tail;
* ``kutta`` — difference of the *downstream* tangential velocities on the upper and lower
  surfaces at matching stations just ahead of the trailing edge: the flow leaves the trailing
  edge smoothly, the condition AERO-01's panel method imposes on its trailing-edge panels.
  It must be directional: equal *speeds* are also satisfied by the circulation-free flow,
  which runs downstream on one side and upstream on the other (a first version of this loss
  compared squared speeds and trained to Gamma ~ 0);
* ``far``   — psi'^2 + (r d(psi')/dr)^2 on r = 8 chords. The true disturbance there is a
  doublet of order 1e-2, so this truncates the domain at a small, known error. Each half
  forbids one way of cheating: psi' = 0 alone admits psi' = (Gamma'/2 pi) ln(r/8), letting the
  network carry circulation the case head does not report; a Robin condition
  r d(psi')/dr + psi' = 0 (exact for a doublet, tried first) admits a *large* doublet, which
  the network used to hide a spurious slow-flow solution behind a shell of residual.
* ``data``  — velocity misfit to the AERO-01 panel solution at sparse field points of the
  training cases (absent from the physics-only ablation).

Every resample draws fresh shapes and incidences uniformly from the family box for the
physics terms, and a fresh subset of training cases and points for the data term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pinncfd.autodiff import grad, laplacian
from pinncfd.external.geometry import inside, naca, panel_solution, surface_samples
from pinncfd.pinn_airfoil_flowfield import family
from pinncfd.pinn_airfoil_flowfield.model import R_FAR, X0, AirfoilPINN

KUTTA_STATIONS = (0.97, 0.985)
RESIDUAL_SCALE = 0.1  # [chord] residuals closer than this to the surface are scaled by (d/0.1)^2
SURFACE_OFFSET = 1e-3  # Kutta points sit this far outside the surface [chord]
CIRCULATION_RADIUS = 1.5  # lift is measured by the circulation on this circle [chord]
MIN_DISTANCE = 0.02  # field reference points keep this far from the surface [chord]


def exterior_points(m, p, t, n: int, rng: np.random.Generator) -> np.ndarray:
    """Collocation points outside one section: 1/2 hugging the surface, 1/4 in a box around
    the section, 1/4 spread log-uniformly in radius from 0.6 to R_FAR chords (equal numbers
    per octave of radius, so no shell of the far field goes unsampled)."""
    out = []
    need = n
    while need > 0:
        n_near, n_box = need // 2, need // 4
        n_disk = need - n_near - n_box
        sp, nrm = surface_samples(m, p, t, n_near, rng)
        d = 1e-3 + 0.6 * rng.random(n_near) ** 3
        near = sp + d[:, None] * nrm
        box = np.column_stack([-0.5 + 2 * rng.random(n_box), -0.6 + 1.2 * rng.random(n_box)])
        r = 0.6 * (R_FAR / 0.6) ** rng.random(n_disk)
        th = 2 * np.pi * rng.random(n_disk)
        disk = np.column_stack([X0 + r * np.cos(th), r * np.sin(th)])
        pts = np.vstack([near, box, disk])
        pts = pts[~inside(m, p, t, pts)]
        out.append(pts)
        need -= len(pts)
    return np.vstack(out)[:n]


def residual_weight(m, p, t, pts: np.ndarray, scale: float | None) -> np.ndarray:
    """min(1, (d / scale)^2), d = distance to the surface: the residual made relative.

    Near the leading edge the velocity changes by O(1) over a few hundredths of a chord, so the
    Laplacian of psi is O(100) there and any small relative error of the network becomes an
    O(1) residual that swamps the mean-square loss; the optimiser then flattens the
    leading-edge flow and distorts the whole field. Scaling the residual by d^2 (its natural
    length scale) measures every point's *relative* error instead. ``scale=None`` disables it.
    """
    if scale is None:
        return np.ones(len(pts))
    from pinncfd.external.geometry import contour

    nodes = contour(m, p, t, 200)
    d = np.min(np.linalg.norm(pts[:, None, :] - nodes[None], axis=2), axis=1)
    return np.minimum(1.0, (d / scale) ** 2)


def kutta_points(m, p, t, h: float = 1e-4) -> np.ndarray:
    """Upper then lower surface points at KUTTA_STATIONS, moved SURFACE_OFFSET outward, with
    their downstream unit tangents, as rows (x, y, tx, ty)."""
    sec = naca(m, p, t)
    xs = np.array(KUTTA_STATIONS)
    xu, yu, xl, yl = sec.surfaces(xs)
    xu2, yu2, xl2, yl2 = sec.surfaces(xs + h)
    rows = []
    for side, (x, y, x2, y2) in enumerate(((xu, yu, xu2, yu2), (xl, yl, xl2, yl2))):
        tx, ty = x2 - x, y2 - y
        norm = np.hypot(tx, ty)
        tx, ty = tx / norm, ty / norm
        # outward normal: tangent rotated +90 deg on the upper surface, -90 deg on the lower
        sgn = 1.0 if side == 0 else -1.0
        nx, ny = -sgn * ty, sgn * tx
        rows.append(np.column_stack([x + SURFACE_OFFSET * nx, y + SURFACE_OFFSET * ny, tx, ty]))
    return np.vstack(rows)


def field_points(m, p, t, n: int, rng: np.random.Generator) -> np.ndarray:
    """Reference points at least MIN_DISTANCE from the surface, where the panel solution is
    converged (its near field is only resolved beyond a few panel lengths)."""
    from pinncfd.external.geometry import contour

    nodes = contour(m, p, t, 400)
    out, need = [], n
    while need > 0:
        cand = exterior_points(m, p, t, 2 * need, rng)
        dist = np.min(np.linalg.norm(cand[:, None, :] - nodes[None], axis=2), axis=1)
        cand = cand[dist >= MIN_DISTANCE]
        out.append(cand[:need])
        need -= len(cand[:need])
    return np.vstack(out)


@dataclass
class Reference:
    """Panel-method reference for a list of cases, flattened for one batched forward pass."""

    cases: list
    field_xy: np.ndarray  # (C * nf, 2)
    field_uv: np.ndarray  # (C * nf, 2)
    surf_xy: np.ndarray  # (C * ns, 2) panel control points
    surf_cp: np.ndarray  # (C * ns,)
    surf_n: np.ndarray  # (C * ns, 2) outward normals
    surf_ds: np.ndarray  # (C * ns,) panel lengths
    cl: np.ndarray  # (C,)
    n_field: int
    n_surf: int

    def q(self, n_per_case: int) -> np.ndarray:
        return np.repeat(np.array([c.params for c in self.cases]), n_per_case, axis=0)


def build_reference(cases, n_field=512, n_panels=400, seed=0) -> Reference:
    rng = np.random.default_rng(seed)
    fxy, fuv, sxy, scp, sn, sds, cl = [], [], [], [], [], [], []
    for c in cases:
        sol = panel_solution(c.m, c.p, c.t, c.alpha, n_panels)
        pts = field_points(c.m, c.p, c.t, n_field, rng)
        u, v = sol.velocity(pts[:, 0], pts[:, 1])
        fxy.append(pts)
        fuv.append(np.column_stack([u, v]))
        sxy.append(sol.control_points)
        scp.append(sol.cp)
        sn.append(sol.normals)
        sds.append(sol.lengths)
        cl.append(sol.cl)
    return Reference(
        list(cases),
        np.vstack(fxy),
        np.vstack(fuv),
        np.vstack(sxy),
        np.concatenate(scp),
        np.vstack(sn),
        np.concatenate(sds),
        np.array(cl),
        n_field,
        n_panels,
    )


def cl_from_cp(cp, normals, ds, alpha) -> float:
    """Lift coefficient from surface pressure on panels (chord 1)."""
    fx = -np.sum(cp * normals[:, 0] * ds)
    fy = -np.sum(cp * normals[:, 1] * ds)
    return float(fy * np.cos(alpha) - fx * np.sin(alpha))


class AirfoilProblem:
    def __init__(
        self,
        model: AirfoilPINN,
        train=None,
        evaluation=None,
        n_cases: int = 32,
        n_interior: int = 384,
        n_body: int = 128,
        n_far: int = 64,
        n_data_cases: int = 16,
        n_data: int = 128,
        weights: dict | None = None,
        use_data: bool = True,
        use_physics: bool = True,
        physics_cases=None,
        residual_scale: float | None = RESIDUAL_SCALE,
        device=None,
        seed: int = 0,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.n_cases, self.n_interior, self.n_body, self.n_far = n_cases, n_interior, n_body, n_far
        self.n_data_cases, self.n_data = n_data_cases, n_data
        self.use_data, self.use_physics = use_data, use_physics
        self.residual_scale = residual_scale
        # physics parameters are drawn from the whole box unless a fixed list is given
        self.physics_params = (
            None if physics_cases is None else np.array([c.params for c in physics_cases])
        )
        physics = ["pde", "body", "tangency", "kutta", "far"]
        terms = (physics if use_physics else []) + (["data"] if use_data else [])
        defaults = {
            "pde": 1.0,
            "body": 10.0,
            "tangency": 1.0,
            "kutta": 10.0,
            "far": 10.0,
            "data": 10.0,
        }
        self.weights = {k: defaults[k] for k in terms}
        self.weights.update({k: v for k, v in (weights or {}).items() if k in terms})
        self.train_cases = list(train if train is not None else family.train_cases())
        if use_data:
            self.train_ref = build_reference(self.train_cases, n_field=4 * n_data, seed=seed + 100)
        self.reference = (
            evaluation if evaluation is not None else build_reference(family.val_cases(), seed=1)
        )
        self.resample(np.random.default_rng(seed))

    @property
    def n_collocation(self) -> int:
        n = 0
        if self.use_physics:
            per_case = self.n_interior + self.n_body + self.n_far + 2 * len(KUTTA_STATIONS)
            n += self.n_cases * per_case
        if self.use_data:
            n += self.n_data_cases * self.n_data
        return n

    def _t(self, a, requires_grad=False):
        t = torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)
        return t.requires_grad_(requires_grad)

    def resample(self, rng: np.random.Generator) -> None:
        if self.use_physics:
            if self.physics_params is None:
                qs = family.uniform(self.n_cases, rng)
            else:
                qs = self.physics_params[rng.integers(0, len(self.physics_params), self.n_cases)]
            xin, win, xb, xk = [], [], [], []
            for m, p, t, _ in qs:
                pts = exterior_points(m, p, t, self.n_interior, rng)
                xin.append(pts)
                win.append(residual_weight(m, p, t, pts, self.residual_scale))
                xb.append(np.hstack(surface_samples(m, p, t, self.n_body, rng)))
                xk.append(kutta_points(m, p, t))
            th = 2 * np.pi * rng.random(self.n_cases * self.n_far)
            xf = np.column_stack([X0 + R_FAR * np.cos(th), R_FAR * np.sin(th)])
            self.xy_in = self._t(np.vstack(xin), True)
            self.w_in = self._t(np.concatenate(win)[:, None])
            self.q_in = self._t(np.repeat(qs, self.n_interior, axis=0))
            xb = np.vstack(xb)
            self.xy_body = self._t(xb[:, :2], True)
            self.n_body_normals = self._t(xb[:, 2:])
            self.q_body = self._t(np.repeat(qs, self.n_body, axis=0))
            self.xy_far = self._t(xf, True)
            self.q_far = self._t(np.repeat(qs, self.n_far, axis=0))
            xk = np.stack(xk)  # (cases, 2 * stations, 4): x, y, tx, ty
            self.xy_kutta = self._t(xk[..., :2].reshape(-1, 2), True)
            self.t_kutta = self._t(xk[..., 2:].reshape(-1, 2))
            self.q_kutta = self._t(np.repeat(qs, 2 * len(KUTTA_STATIONS), axis=0))
        if self.use_data:
            ref = self.train_ref
            idx_cases = rng.choice(len(ref.cases), self.n_data_cases, replace=False)
            rows = np.concatenate(
                [
                    i * ref.n_field + rng.choice(ref.n_field, self.n_data, replace=False)
                    for i in idx_cases
                ]
            )
            q = ref.q(ref.n_field)
            self.xy_data = self._t(ref.field_xy[rows], True)
            self.q_data = self._t(q[rows])
            self.uv_data = self._t(ref.field_uv[rows])

    def losses(self, model) -> dict[str, torch.Tensor]:
        out = {}
        if self.use_physics:
            # the known part is harmonic, so the residual is that of psi' alone
            psi = model.perturbation(self.xy_in, self.q_in)
            out["pde"] = (self.w_in * laplacian(psi, self.xy_in).pow(2)).mean()
            _, psi_b = model.case_scalars(self.q_body)
            out["body"] = (model(self.xy_body, self.q_body) - psi_b).pow(2).mean()
            u, v = model.velocity(self.xy_body, self.q_body)
            vn = u * self.n_body_normals[:, :1] + v * self.n_body_normals[:, 1:]
            out["tangency"] = vn.pow(2).mean()
            u, v = model.velocity(self.xy_kutta, self.q_kutta)
            vt = u * self.t_kutta[:, :1] + v * self.t_kutta[:, 1:]
            vt = vt.view(self.n_cases, 2, len(KUTTA_STATIONS))
            out["kutta"] = (vt[:, 0] - vt[:, 1]).pow(2).mean()
            pf = model.perturbation(self.xy_far, self.q_far)
            g = grad(pf, self.xy_far)
            r_dr = (self.xy_far[:, :1] - X0) * g[:, :1] + self.xy_far[:, 1:2] * g[:, 1:2]
            out["far"] = (pf.pow(2) + r_dr.pow(2)).mean()
        if self.use_data:
            u, v = model.velocity(self.xy_data, self.q_data)
            out["data"] = ((u - self.uv_data[:, :1]) ** 2 + (v - self.uv_data[:, 1:]) ** 2).mean()
        return out

    def predict(self, model, xy: np.ndarray, q: np.ndarray):
        return predict_velocity(model, xy, q, self.device)

    def evaluate(self, model, reference: Reference | None = None, per_case: bool = False):
        return evaluate(model, reference or self.reference, self.device, per_case)


def predict_velocity(model, xy: np.ndarray, q: np.ndarray, device, chunk: int = 32768):
    """Velocities (u, v) at ``xy`` for parameter rows ``q``, as float64 arrays."""
    us, vs = [], []
    for i in range(0, len(xy), chunk):
        t = torch.as_tensor(xy[i : i + chunk], dtype=torch.float32, device=device)
        qt = torch.as_tensor(q[i : i + chunk], dtype=torch.float32, device=device)
        u, v = model.velocity(t.requires_grad_(), qt)
        us.append(u.detach().cpu().double().numpy().ravel())
        vs.append(v.detach().cpu().double().numpy().ravel())
    return np.concatenate(us), np.concatenate(vs)


def surface_cp(model, ref: Reference, device) -> np.ndarray:
    """PINN Cp at the reference's panel control points, (cases, panels)."""
    us, vs = predict_velocity(model, ref.surf_xy, ref.q(ref.n_surf), device)
    return (1.0 - us**2 - vs**2).reshape(len(ref.cases), ref.n_surf)


def lift_from_circulation(model, params: np.ndarray, device, n: int = 256) -> np.ndarray:
    """cl = 2 Gamma (Kutta-Joukowski, chord 1, V = 1) with Gamma the clockwise circulation of the
    predicted velocity on a circle of radius CIRCULATION_RADIUS about (X0, 0), one per row
    (m, p, t, alpha). The periodic trapezoidal rule is spectrally accurate on the circle."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([X0 + CIRCULATION_RADIUS * np.cos(th), CIRCULATION_RADIUS * np.sin(th)])
    xy = np.tile(ring, (len(params), 1))
    u, v = predict_velocity(model, xy, np.repeat(params, n, axis=0), device)
    # counter-clockwise tangent (-sin, cos); clockwise circulation is minus the line integral
    vt = -u * np.tile(np.sin(th), len(params)) + v * np.tile(np.cos(th), len(params))
    gamma_cw = -vt.reshape(len(params), n).mean(axis=1) * 2 * np.pi * CIRCULATION_RADIUS
    return 2.0 * gamma_cw


def evaluate(model, ref: Reference, device, per_case: bool = False):
    """Errors against a panel reference: field velocity, surface Cp and lift."""
    nc = len(ref.cases)
    u, v = predict_velocity(model, ref.field_xy, ref.q(ref.n_field), device)
    duv = (np.column_stack([u, v]) - ref.field_uv).reshape(nc, ref.n_field, 2)
    uv_ref = ref.field_uv.reshape(nc, ref.n_field, 2)
    err = np.linalg.norm(duv, axis=(1, 2)) / np.linalg.norm(uv_ref, axis=(1, 2))
    cp = surface_cp(model, ref, device)
    cp_ref = ref.surf_cp.reshape(nc, ref.n_surf)
    x = ref.surf_xy[:, 0].reshape(nc, ref.n_surf)
    mid = (x > 0.05) & (x < 0.95)
    dcp = cp - cp_ref
    normals = ref.surf_n.reshape(nc, -1, 2)
    ds = ref.surf_ds.reshape(nc, -1)
    cl_p = np.array([cl_from_cp(cp[i], normals[i], ds[i], ref.cases[i].alpha) for i in range(nc)])
    cl_g = lift_from_circulation(model, np.array([c.params for c in ref.cases]), device)
    cases = {
        "error": err,
        "cp_rms": np.sqrt(np.mean(dcp**2, axis=1)),
        "cp_rms_mid": np.sqrt([np.mean(d[k] ** 2) for d, k in zip(dcp, mid, strict=True)]),
        "cp_max_abs": np.abs(dcp).max(axis=1),
        "cl_err_circulation": np.abs(cl_g - ref.cl),
        "cl_err_pressure": np.abs(cl_p - ref.cl),
    }
    if per_case:
        return cases | {"cl_circulation": cl_g, "cl_pressure": cl_p, "cl_ref": ref.cl, "cp": cp}
    out = {k: float(np.mean(v)) for k, v in cases.items()}
    out["error_max"] = float(err.max())
    return out
