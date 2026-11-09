"""Loss terms, collocation and exact-solution metrics for the Joukowski PINN (RES-01 case 2).

The loss terms are those of the airfoil family (``pinn_airfoil_flowfield.problem``), with the
same meaning and default weights, for one section at one incidence and with no data term:
case 2 is physics only.

* ``pde`` — Laplace residual of psi', scaled by min(1, (d / 0.1)^2) next to the surface;
* ``body`` — psi - psi_b on the surface;
* ``tangency`` — normal velocity on the surface;
* ``kutta`` — the jump in downstream tangential velocity across the trailing edge,
  *extrapolated to the edge*. Above and below the edge at x = 0.97 and 0.985 (as in the
  family), the jump Delta(x) is measured and extrapolated linearly in s = sqrt(1 - x) to s = 0,
  and that extrapolated jump is penalised. The family's condition, which penalises Delta at the
  stations themselves, is biased on a *cusped* edge: a loaded cusp keeps a jump that shrinks
  only like sqrt(1 - x), so the exact flow leaves that loss at 1.3e-2 at 4 degrees, while a flow
  with 10 % *less* circulation scores 1.4e-3. The loss is minimised by too little lift. The
  extrapolated jump is 3e-7 for the exact flow and 5e-2 for 90 % of its circulation
  (``tests/test_joukowski_pinn.py`` checks both);
* ``far`` — psi'^2 + (r d(psi')/dr)^2 on r = 8 chords about (0.5, 0).

The reference is the exact flow (:class:`JoukowskiFlow`), so errors can be measured anywhere.
There are two fixed, disjoint evaluation sets with their own seeds: ``"val"`` (what a search
may see) and ``"test"`` (touched once, for reporting). Each has field points at least 0.005
chord off the surface and body points interleaved with the other set's.
"""

from __future__ import annotations

import numpy as np
import torch

from pinncfd.autodiff import grad, laplacian
from pinncfd.external.geometry import JoukowskiSection
from pinncfd.pinn_airfoil_flowfield.model import R_FAR, X0
from pinncfd.pinn_airfoil_flowfield.problem import (
    CIRCULATION_RADIUS,
    KUTTA_STATIONS,
    RESIDUAL_SCALE,
    SURFACE_OFFSET,
    cl_from_cp,
)
from pinncfd.pinn_joukowski.exact import JoukowskiFlow
from pinncfd.pinn_joukowski.model import JoukowskiPINN

EVAL_MIN_DISTANCE = 0.005  # [chord]
SPLIT_SEEDS = {"val": 1001, "test": 2002}
SPLIT_PHASE = {"val": 0.0, "test": 0.5}
TE_SKIP = 0.05  # [rad of circle angle] body points this close to the cusp are left out
DEFAULT_WEIGHTS = {"pde": 1.0, "body": 10.0, "tangency": 1.0, "kutta": 10.0, "far": 10.0}


def distance_to_surface(section: JoukowskiSection, pts: np.ndarray, n: int = 400) -> np.ndarray:
    nodes = section.contour(n)
    return np.min(np.linalg.norm(pts[:, None, :] - nodes[None], axis=2), axis=1)


def exterior_points(section: JoukowskiSection, n: int, rng: np.random.Generator) -> np.ndarray:
    """As the airfoil family's sampler: 1/2 hugging the surface, 1/4 in a box around the
    section, 1/4 log-uniform in radius from 0.6 to R_FAR chords about (X0, 0)."""
    out, need = [], n
    while need > 0:
        n_near, n_box = need // 2, need // 4
        n_disk = need - n_near - n_box
        sp, nrm = section.surface_samples(n_near, rng)
        d = 1e-3 + 0.6 * rng.random(n_near) ** 3
        near = sp + d[:, None] * nrm
        box = np.column_stack([-0.5 + 2 * rng.random(n_box), -0.6 + 1.2 * rng.random(n_box)])
        r = 0.6 * (R_FAR / 0.6) ** rng.random(n_disk)
        th = 2 * np.pi * rng.random(n_disk)
        disk = np.column_stack([X0 + r * np.cos(th), r * np.sin(th)])
        pts = np.vstack([near, box, disk])
        pts = pts[~section.inside(pts)]
        out.append(pts)
        need -= len(pts)
    return np.vstack(out)[:n]


KUTTA_S = np.sqrt(1 - np.array(KUTTA_STATIONS))


def kutta_points(section: JoukowskiSection, h: float = 1e-5) -> np.ndarray:
    """Upper then lower surface points at KUTTA_STATIONS, moved SURFACE_OFFSET outward, with
    their downstream unit tangents, as rows (x, y, tx, ty)."""
    xs = np.array(KUTTA_STATIONS)
    xu, yu, xl, yl = section.surfaces(xs)
    xu2, yu2, xl2, yl2 = section.surfaces(xs + h)
    rows = []
    for side, (x, y, x2, y2) in enumerate(((xu, yu, xu2, yu2), (xl, yl, xl2, yl2))):
        tx, ty = x2 - x, y2 - y
        norm = np.hypot(tx, ty)
        tx, ty = tx / norm, ty / norm
        sgn = 1.0 if side == 0 else -1.0
        nx, ny = -sgn * ty, sgn * tx
        rows.append(np.column_stack([x + SURFACE_OFFSET * nx, y + SURFACE_OFFSET * ny, tx, ty]))
    return np.vstack(rows)


def trailing_edge_jump(vt_upper, vt_lower):
    """Jump of the downstream tangential velocity, extrapolated linearly in s = sqrt(1 - x)
    from the two Kutta stations to the edge (s = 0). Works on arrays and tensors."""
    d1, d2 = vt_upper[..., 0] - vt_lower[..., 0], vt_upper[..., 1] - vt_lower[..., 1]
    s1, s2 = KUTTA_S
    return (s2 * d1 - s1 * d2) / (s2 - s1)


def eval_points(section: JoukowskiSection, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out, need = [], n
    while need > 0:
        cand = exterior_points(section, 2 * need, rng)
        cand = cand[distance_to_surface(section, cand) >= EVAL_MIN_DISTANCE][:need]
        out.append(cand)
        need -= len(cand)
    return np.vstack(out)


def ring(n: int = 256) -> tuple[np.ndarray, np.ndarray]:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.column_stack([X0 + CIRCULATION_RADIUS * np.cos(th), CIRCULATION_RADIUS * np.sin(th)])
    return xy, th


def circulation_lift(u, v, th) -> float:
    """cl = 2 Gamma, Gamma the clockwise circulation on the ring (chord 1, V = 1)."""
    vt = -u * np.sin(th) + v * np.cos(th)
    return float(2 * -np.mean(vt) * 2 * np.pi * CIRCULATION_RADIUS)


def score(field_uv, field_ref, cp, cp_ref, x_body, cl_circ, cl_pres, cl_pres_ref, cl_exact):
    """Error metrics from predicted and reference arrays (pure NumPy, so testable exactly)."""
    duv = np.asarray(field_uv) - np.asarray(field_ref)
    dcp = np.asarray(cp) - np.asarray(cp_ref)
    mid = (x_body > 0.05) & (x_body < 0.95)
    return {
        "error": float(np.linalg.norm(duv) / np.linalg.norm(field_ref)),
        "cp_rms": float(np.sqrt(np.mean(dcp**2))),
        "cp_rms_mid": float(np.sqrt(np.mean(dcp[mid] ** 2))),
        "cp_max_abs": float(np.abs(dcp).max()),
        "cl_err_circulation": float(abs(cl_circ - cl_exact)),
        "cl_err_pressure": float(abs(cl_pres - cl_pres_ref)),
    }


class JoukowskiProblem:
    def __init__(
        self,
        model: JoukowskiPINN,
        n_interior: int = 2048,
        n_body: int = 512,
        n_far: int = 256,
        weights: dict | None = None,
        residual_scale: float | None = RESIDUAL_SCALE,
        n_eval: int = 2048,
        n_eval_body: int = 400,
        split: str = "val",
        device=None,
        seed: int = 0,
        dtype=torch.float32,
    ):
        self.model = model
        self.section = model.section
        self.flow = JoukowskiFlow(self.section)
        self.alpha = model.alpha
        self.device = device or next(model.parameters()).device
        self.dtype = dtype
        self.n_interior, self.n_body, self.n_far = n_interior, n_body, n_far
        self.residual_scale = residual_scale
        self.weights = dict(DEFAULT_WEIGHTS)
        self.weights.update({k: v for k, v in (weights or {}).items() if k in DEFAULT_WEIGHTS})
        self.split = split
        xk = kutta_points(self.section)
        self.xy_kutta = self._t(xk[:, :2], True)
        self.t_kutta = self._t(xk[:, 2:])
        self.refs = {s: self._reference(s, n_eval, n_eval_body) for s in SPLIT_SEEDS}
        self.resample(np.random.default_rng(seed))

    # ---- reference -------------------------------------------------------------------------
    def _reference(self, split: str, n_eval: int, n_body: int) -> dict:
        flow, a = self.flow, self.alpha
        xy = eval_points(self.section, n_eval, SPLIT_SEEDS[split])
        bx, by, _ = flow.body_points(n_body, skip_te=TE_SKIP, phase=SPLIT_PHASE[split])
        nodes = self.section.contour(400)
        mids = (nodes[:-1] + nodes[1:]) / 2
        d = np.diff(nodes, axis=0)
        ds = np.linalg.norm(d, axis=1)
        normals = np.column_stack([-d[:, 1], d[:, 0]]) / ds[:, None]
        return {
            "field_xy": xy,
            "field_uv": np.column_stack(flow.velocity(xy[:, 0], xy[:, 1], a)),
            "body_xy": np.column_stack([bx, by]),
            "body_cp": flow.cp(bx, by, a),
            "panel_xy": mids,
            "panel_n": normals,
            "panel_ds": ds,
            # the exact Cp integrated with the same panels, so quadrature error cancels
            "cl_pressure_ref": cl_from_cp(flow.cp(mids[:, 0], mids[:, 1], a), normals, ds, a),
            "cl": flow.cl(a),
        }

    @property
    def n_collocation(self) -> int:
        return self.n_interior + self.n_body + self.n_far + len(self.xy_kutta)

    def _t(self, a, requires_grad=False):
        t = torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device)
        return t.requires_grad_(requires_grad)

    # ---- collocation and losses ------------------------------------------------------------
    def resample(self, rng: np.random.Generator) -> None:
        sec = self.section
        pts = exterior_points(sec, self.n_interior, rng)
        if self.residual_scale is None:
            w = np.ones(len(pts))
        else:
            w = np.minimum(1.0, (distance_to_surface(sec, pts, 200) / self.residual_scale) ** 2)
        self.xy_in = self._t(pts, True)
        self.w_in = self._t(w[:, None])
        xb, nb = sec.surface_samples(self.n_body, rng)
        self.xy_body = self._t(xb, True)
        self.n_body_normals = self._t(nb)
        th = 2 * np.pi * rng.random(self.n_far)
        self.xy_far = self._t(np.column_stack([X0 + R_FAR * np.cos(th), R_FAR * np.sin(th)]), True)

    def physics_losses(self, model) -> dict[str, torch.Tensor]:
        psi = model.perturbation(self.xy_in)
        out = {"pde": (self.w_in * laplacian(psi, self.xy_in).pow(2)).mean()}
        out["body"] = (model(self.xy_body) - model.psi_b).pow(2).mean()
        u, v = model.velocity(self.xy_body)
        vn = u * self.n_body_normals[:, :1] + v * self.n_body_normals[:, 1:]
        out["tangency"] = vn.pow(2).mean()
        u, v = model.velocity(self.xy_kutta)
        vt = (u * self.t_kutta[:, :1] + v * self.t_kutta[:, 1:]).view(2, len(KUTTA_STATIONS))
        out["kutta"] = trailing_edge_jump(vt[0], vt[1]).pow(2)
        pf = model.perturbation(self.xy_far)
        g = grad(pf, self.xy_far)
        r_dr = (self.xy_far[:, :1] - X0) * g[:, :1] + self.xy_far[:, 1:2] * g[:, 1:2]
        out["far"] = (pf.pow(2) + r_dr.pow(2)).mean()
        return out

    def losses(self, model) -> dict[str, torch.Tensor]:
        return self.physics_losses(model)

    # ---- evaluation ------------------------------------------------------------------------
    def predict(self, model, xy: np.ndarray, chunk: int = 32768):
        """Velocities (u, v) at ``xy`` as float64 arrays."""
        us, vs = [], []
        for i in range(0, len(xy), chunk):
            u, v = model.velocity(self._t(xy[i : i + chunk], True))
            us.append(u.detach().cpu().double().numpy().ravel())
            vs.append(v.detach().cpu().double().numpy().ravel())
        return np.concatenate(us), np.concatenate(vs)

    def evaluate(self, model, split: str | None = None) -> dict[str, float]:
        ref = self.refs[split or self.split]
        a = self.alpha
        u, v = self.predict(model, ref["field_xy"])
        ub, vb = self.predict(model, ref["body_xy"])
        up, vp = self.predict(model, ref["panel_xy"])
        cl_pres = cl_from_cp(1 - up**2 - vp**2, ref["panel_n"], ref["panel_ds"], a)
        rxy, th = ring()
        ur, vr = self.predict(model, rxy)
        out = score(
            np.column_stack([u, v]),
            ref["field_uv"],
            1 - ub**2 - vb**2,
            ref["body_cp"],
            ref["body_xy"][:, 0],
            circulation_lift(ur, vr, th),
            cl_pres,
            ref["cl_pressure_ref"],
            ref["cl"],
        )
        out["gamma_err"] = abs(float(model.gamma.detach()) - self.flow.circulation(a))
        return out
