"""RES-01 inverse problem: incidence and speed from surface pressure taps (air data)."""

from pinncfd import benchmark
from pinncfd.inverse_airdata.model import AirDataPINN
from pinncfd.inverse_airdata.problem import AirDataProblem

__all__ = ["AirDataPINN", "AirDataProblem"]


def _build_airdata(
    genome,
    seed,
    device,
    alpha: float = 0.0698131700797732,  # 4 degrees
    u: float = 1.15,
    n_taps: int = 8,
    noise: float = 0.0,
    draw: int = 0,
    noise_split: str = "val",
    alpha0: float = 0.0,
    u0: float = 1.0,
    **problem_kw,
):
    model = AirDataPINN(
        alpha0, u0, genome.get("features", "plain"), benchmark.mlp_from(genome, seed)
    )
    model = model.to(device)
    kw = dict(n_interior=genome["n_interior"], weights=benchmark.weights_from(genome)) | problem_kw
    problem = AirDataProblem(
        model, alpha, u, n_taps, noise, draw, noise_split, device=device, seed=seed, **kw
    )
    return model, problem


benchmark.BUILDERS["airdata"] = _build_airdata
