# Neuroevolutionary search over architecture and loss weights

Code: [`src/pinncfd/neuroevolution_search/`](../src/pinncfd/neuroevolution_search/) ·
Lesson: [Lesson 3](lessons/03-neuroevolution.md) ·
Notebook: [`03_neuroevolution.ipynb`](../notebooks/03_neuroevolution.ipynb) ·
Results: [`validation/results/neuroevolution.md`](../validation/results/neuroevolution.md)

## The genome

A flat set of twelve independent genes ([`genome.py`](../src/pinncfd/neuroevolution_search/genome.py)):

| Group | Genes | Domain |
|---|---|---|
| Architecture | depth; width; activation; Fourier features and their scale | {3, 4, 5, 6}; {32, 48, 64, 96}; {tanh, sin, SiLU}; {0, 16, 32} and U(0.25, 2) |
| Optimisation | Adam learning rate; interior points per shape | log-U(3e-4, 3e-3); {192, 384, 768} |
| Loss weights | $\lambda$ for body, tangency, Kutta, far field and data | log-U(0.1, 100) each |

The PDE weight stays 1, since only ratios matter. Weights are log-uniform because their effect
is multiplicative. Uniform crossover takes each gene from either parent; mutation redraws each
gene from its domain with probability 0.2. Both stay in the domain by construction.

## NSGA-II

Two objectives are minimised: the validation velocity error after a proxy budget, and the
parameter count. $a$ dominates $b$ when it is no worse on both and better on one. NSGA-II
(Deb et al. 2002) keeps the non-dominated set as it evolves:

1. Non-dominated sorting splits the population into fronts $F_0, F_1, \dots$
2. Within a front, the crowding distance sums the normalised gaps to each genome's neighbours
   on every objective; the ends of a front get infinite distance.
3. Binary tournaments pick parents: lower front first, then larger crowding distance.
4. Parents and children are pooled; the best `population_size` survive by front, and the last
   front that only partly fits is cut by crowding distance.

The implementation is ported from the RUL project (`src/rul/evolution/`, commit `addc209`),
pure NumPy. The search state, every genome trained with its fitness, is a JSON checkpoint from
which a run resumes. Tests check the sorting and crowding on hand-checkable fronts, and that
genomes stay in their domains.

## The run

Population 8, 5 generations, 40 genomes. Generation 0 is the hand-tuned Phase 2 genome plus
seven random ones. Proxy budget: 400 Adam steps and 100 L-BFGS iterations, about a minute
each, one seed per genome. The front's most accurate genome is then retrained at the full
Phase 2 budget, 4000 + 1000 steps, over three seeds.

## Results

| | Phase 2 (hand-tuned) | Phase 3 champion |
|---|---|---|
| Genome | 5×64 tanh, 384 points, λ = 10/1/10/10/10 | 6×32 SiLU, 192 points, λ = 5.4/6.6/15/3.2/64 |
| Parameters | 18,499 | 6,851 |
| Proxy error | 0.058 | 0.024 |
| Validation error, full budget | 1.17 % ± 0.3 % | 0.72 % ± 0.11 % |
| Test velocity error | 1.16 % | 0.78 % ± 0.17 % |
| Test cl error (circulation) | 0.121 | 0.037 |
| Training time | 575 ± 30 s | 653 ± 5.5 s |
| Steps to 1e-2 | 5000 (one seed in three) | 4500–4750 |

λ is listed as body/tangency/Kutta/far/data.

Search cost: 40.1 min, plus 32.7 min for the three retrainings, on the RTX 2060 at 1000 MHz.
The proxy overstated the gain: 2.4× at the proxy budget became 1.6× at full budget. All 14
genomes with Fourier features scored 0.115 or worse. A higher data weight correlated with lower
error, but genes are correlated by descent and each genome had one seed, so these patterns are
leads, not findings.

## References

- K. Deb, A. Pratap, S. Agarwal & T. Meyarivan, *IEEE Trans. Evol. Comput.* 6, 2002.
- K. O. Stanley & R. Miikkulainen, *Evol. Comput.* 10, 2002.
- T. Elsken, J. H. Metzen & F. Hutter, *JMLR* 20, 2019.
- L. Li et al., "Hyperband", *JMLR* 18, 2018.
