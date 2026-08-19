<div align="center">
  <img src="logo.png" width="350" />
  <h1>SOVEREIGN</h1>
  <p><strong>S</strong>tochastic <strong>O</strong>rder-driven <strong>V</strong>olatility <strong>E</strong>ngine with <strong>R</strong>ecursive <strong>E</strong>ndogenous <strong>I</strong>nstability, <strong>G</strong>enerated <strong>N</strong>umerically</p>

  <a href="https://drive.google.com/drive/u/0/folders/1VdiBC7ijt8NYdPndfwThRduXn03N2m4R"><strong>📄 Read the Whitepaper (Mathematical Specification)</strong></a>
  <br /><br />
  <img src="https://img.shields.io/badge/C%2B%2B-20-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="C++20" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Qt-PyQt6-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="PyQt6" />
  <img src="https://img.shields.io/badge/OpenGL-3D_Telemetry-FFFFFF?style=for-the-badge&logo=opengl&logoColor=black" alt="OpenGL" />
  <img src="https://img.shields.io/badge/Math-Eigen3-red?style=for-the-badge" alt="Eigen3" />
</div>

> A C++20 multi-asset limit order book simulation engine that models endogenous flash crashes via rough volatility, multivariate Hawkes processes, and robust market making under Knightian ambiguity.

**Version:** 0.1.0 | **Standard:** C++20 | **Author:** keykyrios (Mitrajit Ghorui)

---

## Architecture

SOVEREIGN is a header-only simulation engine (22 `.hpp` files under `include/sovereign/`) compiled as a static library `sovereign_lib`. Entry point: `apps/main.cpp`.

### 7-Layer Pipeline

Each tick (default Δt = 10⁻⁴) executes in strict causal order:

| Layer | Engine Class(es) | File(s) | Description |
|-------|-------------------|---------|-------------|
| 1 | `RoughVolEngine`, `CGMYEngine`, `RegimeEngine` | `price/rough_vol.hpp`, `price/levy_jumps.hpp`, `price/regime.hpp` | 12-factor Markovian fBm + CIR-subordinated CGMY jumps + 5-state HMM |
| 2 | `HawkesEngine` | `hawkes/multivariate.hpp` | 10-component sum-of-exp Ogata thinning, 10⁶ event cap |
| 3 | `LOBEngine` (`OrderBook`) | `orderbook/lob.hpp` | Eigen::VectorXi book, iceberg orders, √-concave impact |
| 4 | `MarketMakerEngine` | `market_maker/robust_control.hpp` | 5 ghost MM agents, HJB PDE on background thread |
| 5 | `RuinEngine` | `ruin/gerber_shiu.hpp` | Cramér-Lundberg + Gerber-Shiu IDE (Picard, 200-pt grid) |
| 6 | `CorrelationEngine`, `GraphEngine`, `SpectralEngine`, `ContagionEngine` | `topology/*.hpp` | EWMA → MP cleaning → Higham → MST/PMFG → Fiedler + Brandes → Laplacian diffusion |
| 7 | `PersistenceEngine`, `LandscapeEngine` | `tda/*.hpp` | Native Z₂ VR complex → TRI + L₁ landscape + Wasserstein-2 |

**Threading:** Layers 1–5 run on the main thread. Layers 6–7 run on a dedicated background thread via double-buffered `SimulationState` snapshots. HJB solving runs on a third thread. Synchronization via `std::mutex` + `std::condition_variable`.

### Feedback Loops (Implemented)

```
RuinEngine (Ψᵢ > 0.5) → HawkesEngine baseline modulation (1.0 + 0.5·max(Ψ-0.5, 0))
RuinEngine (Γ vector, contagion-blended) → HawkesEngine (not local actuarial ruin)
RuinEngine (Γ vector) → ContagionEngine (implicit Euler Laplacian diffusion)
Topology results → HawkesEngine cross-excitation (α_ij *= exp(-0.5·d_graph))
TRI(t) → θ(t) = θ₀·exp(α·TRI)  [Layer 7 → Layer 4, θ₀=0.5, α=0.1, clamped [0.1,10.0]]
```

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **Eigen3** | ≥ 3.4 | Linear algebra, vectorized matrix ops |
| **nlohmann/json** | any | Config deserialization (`NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT`) |
| **Boost** | ≥ 1.74 | `boost::asio` for TCP telemetry socket |
| **OpenMP** | optional | `#pragma omp parallel for` in price, LOB, impact loops |
| **GUDHI** | optional | Persistent homology (available via `SOVEREIGN_USE_GUDHI=ON` — see Building) |

---

## Building

### Prerequisites

**Linux (apt):**
```bash
sudo apt install libeigen3-dev nlohmann-json3-dev libboost-all-dev
```

**Windows (MSYS2/UCRT64):**
```bash
pacman -S mingw-w64-ucrt-x86_64-eigen3 mingw-w64-ucrt-x86_64-nlohmann-json mingw-w64-ucrt-x86_64-boost
```

**Windows (vcpkg):**
```bash
vcpkg install eigen3 nlohmann-json boost
```

### Build Commands

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

**CMake Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `SOVEREIGN_USE_OPENMP` | ON | Enable OpenMP parallelism |
| `SOVEREIGN_USE_GUDHI` | OFF | Enable GUDHI persistent cohomology backend (requires GUDHI installed) |
| `SOVEREIGN_BUILD_TESTS` | OFF | Build unit tests |

---

## Running

```bash
./sovereign                              # Default: 50 assets, dt=1e-4, T=1.0
./sovereign --config default_config.json  # Load params from JSON
./sovereign --assets 100 --dt 5e-5       # CLI overrides (applied AFTER config load)
./sovereign --seed 12345                 # Reproducible RNG
./sovereign --save-config out.json       # Dump effective config
```

### CLI Arguments

| Flag | Description |
|------|-------------|
| `--config <path>` | Load configuration from JSON file |
| `--assets <N>` | Number of assets (default: 50) |
| `--steps <N>` | Override number of timesteps |
| `--dt <val>` | Timestep size (default: 1e-4) |
| `--seed <val>` | RNG seed (default: 42) |
| `--save-config <path>` | Save effective config to JSON |

---

## Telemetry & Dashboard

The engine streams state snapshots as **length-prefixed JSON** over TCP to `127.0.0.1:8080`:

- **Protocol:** `[4-byte LE uint32 length][JSON payload]`
- **Decimation:** Every 10 ticks (`flush_interval_ = 10`)
- **Serialization:** Direct `std::ostringstream` — NOT nlohmann/json AST
- **Transport:** `boost::asio` TCP client with async write via `std::async`
- **Fault tolerance:** Silent reconnect if dashboard is not running

The dashboard (`sovereign_dashboard.py`) is a Python TCP server rendering via `pyqtgraph.opengl`.

### Telemetry Payload Fields

Per-asset: `price`, `log_price`, `vol`, `variance`, `hurst`, `jump`, `lob_impact`, `regime`, `return1`, `cum_ret`, `hawkes_intensity[50]`, `lob{bid_vol, ask_vol, bid_price, ask_price}[20]`, `mm_spread`, `mm_inventory`, `surplus`, `ruin`, `gerber_shiu`.

Global: `correlation[N×N]`, `raw_correlation[N×N]`, `eigenvalues[N]`, `fiedler`, `clustering`, `betweenness[N]`, `degree[N]`, `tri`, `wasserstein`, `l1`, `l2`, `ruin_vector[N]`, `distance[N×N]`.

---

## Key Implementation Details

### Rough Bergomi (Layer 1)
- **12-factor Markovian approximation** of Volterra kernel K(t) = √(2H)·t^(H-1/2)
- Decay rates λₖ log-spaced from 0.01 to 10,000
- Weights: c_k = √(2H)·λ_k^(γ-1)·Δlog(λ)/Γ(γ) where γ = 1/2 - H (Abi Jaber & El Euch 2019)
- Exact OU diffusion coefficient per factor (no Euler discretization error)
- **Stochastic Hurst** OU process with reflection at [0.01, 0.99]
- **Systemic market factor:** β=0.5 common factor injected before Cholesky

### CGMY Jumps (Layer 1)
- Truncated compound Poisson (|x| ≥ 0.01) + Gaussian residual
- **Alfonsi implicit CIR** subordinator with dynamic Feller enforcement
- Full Laplace exponent compensator ψ(-1) for martingale property
- **Systemic jump:** C_systemic = 0.5C, each asset gets 0.5×J_systemic

### Hawkes (Layer 2)
- **N×5×10 dimensional** intensity (5 event types × 10 depth levels)
- 10-component sum-of-exponentials with O(1) recursive update
- **Power-law weights:** α_m ∝ β_m^{-ε} (ε=0.5, approximates φ(t)=(1+t/β)^{-(1+ε)})
- Mark-weighted excitation: w = clamp(size/11, 0.1, 3.0)
- **Dykstra spectral radius projection** (ρ(R) ≤ 0.95)
- Zipf power-law heterogeneity: cap_scale = 1/√(i+1)

### LOB (Layer 3)
- `Eigen::VectorXi` bid/ask volumes (M=500 levels per side)
- Iceberg orders with **latency-gated revelation** (~5μs)
- OU mid-price drift with tick-snapping (κ=100.0, tight tracking during flash crashes)
- **√-concave impact:** ∑ sign(imbalance)·√|imbalance|·1/(1+d) over top 20 levels

### Market Maker (Layer 4)
- 5 **ghost agents** per asset (no physical LOB orders)
- Closed-form spread: δ* = (γ+θ(t))σ²|I|/2 + 2Ψ + floor, **θ(t)=θ₀·exp(α·TRI)** dynamic
- Background HJB on 30×30 grid, dp=0.01 unified (CFL-adaptive FTCS, dedicated thread)
- Avellaneda-Stoikov fill model: λ = 100·exp(-kδ - η|I|)
- Self-Match Prevention (SMP)

### Ruin (Layer 5)
- Premium = spread × 200 fills/time
- Claims on LOB stress (|impact|×10⁴ > 10.0) + Poisson background (0.05/yr)
- Cramér-Lundberg ruin probability with exponential claims
- Gerber-Shiu IDE: Gauss-Seidel iteration, 200-point grid, every 200 steps
- Contagion blending via exponential decay (rate 5.0) to local ruin probability

### Topology (Layer 6)
- EWMA α=0.005 with adaptive warm-up (first 2N samples)
- Marčenko-Pastur cleaning with effective N_eff = min(2/α, t), σ² from bottom-half spectrum
- Higham alternating projections (50 iterations) for nearest correlation matrix
- Prim's MST O(N²) + PMFG with Euler planarity filter (E≤3V-6) + common-neighbor face check
- Fiedler eigenvalue, Brandes betweenness, local clustering
- Implicit Euler Laplacian contagion diffusion (D=0.1) with unified exp(-√(2(1-ρ))) weights

### TDA (Layer 7)
- Vietoris-Rips filtration up to tetrahedra, ε_max = 2.0
- Native Z₂ boundary matrix reduction with O(1) hash map face lookup
- TRI = Σ (d-b)^p / (1+b), p=2.0 — heuristic, NOT L₁ norm
- L^p landscape norms via trapezoidal quadrature
- Greedy O(n²) Wasserstein-2 on H₁ pairs (upper bound)

---

## Performance

- **RNG:** Xoshiro256** with Box-Muller (cached spare), alignas(64)
- **Thread-local RNGs:** Forked via Vigna's 2¹²⁸ jump polynomial
- **Subnormal prevention:** FTZ + DAZ on x86_64
- **Eigen threading:** Disabled (`setNbThreads(1)`) to prevent OMP oversubscription
- **Event buffer:** Pre-allocated 4096 entries, reused per tick
- **3-thread architecture:** Main tick loop + topology worker + HJB solver

---

## Known Issues & Technical Debt

### Fixed in Current Version

1. **Markovian kernel weights:** Corrected exponent from λ^γ to λ^(γ-1) per Abi Jaber & El Euch (2019).
2. **CIR Alfonsi scheme:** Ito correction now uses `cfg_.cir_lambda` consistently.
3. **CGMY small-jump mean:** Corrected to use `1-Y` exponent for Y∈(1,2) regime.
4. **RMT σ² estimation:** Fixed to use bottom-half (noise) eigenvalues, not top-half (signal).
5. **Gerber-Shiu IDE signs:** Corrected the IDE RHS — conv and ω are positive contributions.
6. **Persistence face lookup:** O(M²) linear scan replaced with O(1) hash map.
7. **Hawkes lambda_bar floor:** Lowered from 1.0 to 1e-6 to prevent wasted proposals.
8. **Zipf cap_scale preservation:** `modulate_by_graph()` now preserves power-law heterogeneity.
9. **Contagion/Spectral Laplacian unification:** Both now use exp(-√(2(1-ρ))) weighting.
10. **Ruin cwiseMax ratchet:** Removed — exponential blend governs onset and recovery.
11. **Distance matrix initialization:** Now arccos(identity) instead of all-zeros.
12. **Landscape quadrature:** Trapezoidal rule (half-weight endpoints) instead of Riemann sum.
13. **MLMC variance estimator:** Unbiased (/(n-1)) + Welford-style incremental update.
14. **GUDHI Build Bug:** Fixed `CMakeLists.txt` to correctly define `SOVEREIGN_HAS_GUDHI`.
15. **TRI → θ Feedback:** Connected the topological risk index to the market maker's ambiguity aversion.
16. **PMFG Planarity:** Added Euler-formula planarity filter and common-neighbor checks.
17. **Hawkes Weights:** Corrected uniform α_m to true power-law weights α_m ∝ β_m^{-ε}.
18. **Variance Bias:** Added missing cross-factor covariances to the exact martingale compensator.
19. **Dead Code Cleanup:** Removed unused `VolterraFBM`, `FBMGenerator`, `n_hybrid_steps`, and `kernels.hpp`.
20. **HJB Grid Consistency:** Unified background worker and engine grid spacing to `dp=0.01`.
21. **Ruin Signal Path:** Clarified that Hawkes feedback reads contagion-blended ruin, not local actuarial ruin.
22. **LOB Price Lag:** Increased OU tracking speed from 1.0 to 100.0 for instant tracking during flash crashes.

### Remaining Issues

1. **Ghost Market Makers:** MMs evaluate spreads and simulate fills internally but never place physical limit orders into the LOB.
2. **Orphaned MLMC:** `mc/mlmc.hpp` contains a complete Giles (2008) multi-level Monte Carlo implementation but is never called by the engine.
3. **No Native Calibration:** All parameters come from `default_config.json`. No MLE/GMM calibration pipeline exists.
4. **Wasserstein Upper Bound:** The greedy O(n²) matching is an upper bound on optimal transport, not exact.

---

## Configuration

All parameters live in `config.hpp` as nested structs with `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT`. Key defaults:

| Parameter | Default | Location |
|-----------|---------|----------|
| `n_assets` | 50 | `UniverseConfig` |
| `T` | 1.0 | `SimulationConfig` |
| `dt` | 1e-4 | `SimulationConfig` |
| `seed` | 42 | `SimulationConfig` |
| `hurst` | 0.10 | `RoughVolConfig` |
| `eta` | 0.35 | `RoughVolConfig` |
| `rho` | -0.90 | `RoughVolConfig` |
| `xi_0` | 0.04 | `RoughVolConfig` |
| `C, G, M, Y` | 1.0, 5.0, 10.0, 1.5 | `LevyConfig` |
| `n_order_types` | 5 | `HawkesConfig` |
| `n_depth_levels` | 10 | `HawkesConfig` |
| `base_intensity` | 10.0 | `HawkesConfig` |
| `max_spectral_radius` | 0.95 | `HawkesConfig` |
| `n_levels` | 500 | `LOBConfig` |
| `n_market_makers` | 5 | `MarketMakerConfig` |
| `gamma` | 2.0 | `MarketMakerConfig` |
| `theta` | 0.5 | `MarketMakerConfig` (base; dynamic θ(t)=θ₀·exp(α·TRI) at runtime) |
| `tri_alpha` | 0.1 | `MarketMakerConfig` (TRI→θ coupling strength) |
| `ewma_alpha` | 0.005 | `TopologyConfig` |
| `max_dimension` | 2 | `TDAConfig` |
| `max_filtration` | 2.0 | `TDAConfig` |
| `tri_weight_power` | 2.0 | `TDAConfig` |

---

## File Structure

```
SOVEREIGN/
├── CMakeLists.txt                    # Build system (C++20, Eigen3, nlohmann/json, Boost)
├── apps/
│   └── main.cpp                      # Entry point with CLI parsing
├── src/
│   └── config.cpp                    # SimulationConfig::from_json/to_json
├── include/sovereign/
│   ├── config.hpp                    # All config structs (214 lines)
│   ├── engine.hpp                    # Main orchestrator (376 lines)
│   ├── core/
│   │   ├── clock.hpp                 # SimulationClock + Event priority queue
│   │   ├── random.hpp                # Xoshiro256** (dead FBMGenerator/VolterraFBM removed)
│   │   └── state.hpp                 # AssetState, SimulationState
│   ├── price/
│   │   ├── rough_vol.hpp             # MarkovianFBM + RoughVolEngine
│   │   ├── levy_jumps.hpp            # CGMYEngine + CIR subordinator
│   │   └── regime.hpp                # RegimeEngine (5-state HMM)
│   ├── hawkes/
│   │   ├── multivariate.hpp          # HawkesEngine (N×5×10, power-law weights)
│   │   ├── kernels.hpp               # Empty stub (dead code removed)
│   │   └── stability.hpp             # Dykstra spectral radius projector
│   ├── orderbook/
│   │   └── lob.hpp                   # OrderBook + LOBEngine
│   ├── market_maker/
│   │   └── robust_control.hpp        # MarketMakerEngine + HJB PDE
│   ├── ruin/
│   │   └── gerber_shiu.hpp           # RuinEngine + Cramér-Lundberg + Picard IDE
│   ├── topology/
│   │   ├── correlation.hpp           # EWMA + RMT + Higham
│   │   ├── graphs.hpp                # Prim MST + PMFG (Euler planarity filter)
│   │   ├── spectral.hpp              # Fiedler + Brandes + clustering
│   │   └── contagion.hpp             # Implicit Euler Laplacian diffusion
│   ├── tda/
│   │   ├── persistence.hpp           # Z₂ boundary matrix reduction (+ GUDHI path)
│   │   └── landscapes.hpp            # Landscapes + TRI + Wasserstein-2
│   ├── mc/
│   │   └── mlmc.hpp                  # MLMC (orphaned — not called by engine)
│   └── viz/
│       └── telemetry.hpp             # TCP length-prefixed JSON telemetry
├── default_config.json               # Default simulation parameters
└── sovereign_dashboard.py            # Python PyQtGraph real-time dashboard
```

---

## License

MIT
