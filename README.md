<div align="center">
  <img src="logo.png" width="350" />
  <h1>SOVEREIGN</h1>
  <p><strong>S</strong>tochastic <strong>O</strong>rder-driven <strong>V</strong>olatility <strong>E</strong>ngine with <strong>R</strong>ecursive <strong>E</strong>ndogenous <strong>I</strong>nstability, <strong>G</strong>enerated <strong>N</strong>umerically</p>

  <a href="https://drive.google.com/file/d/1RrAC7_fYjStJ_nXX5hUgOGOSytEOPVxq/view?usp=sharing"><strong>📄 Read the Whitepaper (Mathematical Specification)</strong></a>
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
RuinEngine (Γ vector) → ContagionEngine (implicit Euler Laplacian diffusion)
Topology results → HawkesEngine cross-excitation (α_ij *= exp(-0.5·d_graph))
```

### Feedback Loops (NOT Implemented)

```
TRI(t) → θ(t) = θ₀·exp(α·TRI)  [Layer 7 → Layer 4 ambiguity coupling — DESIGN TARGET ONLY]
```

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **Eigen3** | ≥ 3.4 | Linear algebra, vectorized matrix ops |
| **nlohmann/json** | any | Config deserialization (`NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT`) |
| **Boost** | ≥ 1.74 | `boost::asio` for TCP telemetry socket |
| **OpenMP** | optional | `#pragma omp parallel for` in price, LOB, impact loops |
| **GUDHI** | optional | Persistent homology (currently **unreachable** — see Known Issues) |

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
| `SOVEREIGN_USE_GUDHI` | OFF | Enable GUDHI (currently broken — see Known Issues) |
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
- Mark-weighted excitation: w = clamp(size/11, 0.1, 3.0)
- **Dykstra spectral radius projection** (ρ(R) ≤ 0.95)
- Zipf power-law heterogeneity: cap_scale = 1/√(i+1)

### LOB (Layer 3)
- `Eigen::VectorXi` bid/ask volumes (M=500 levels per side)
- Iceberg orders with **latency-gated revelation** (~5μs)
- OU mid-price drift with tick-snapping (Regulation NMS sub-penny)
- **√-concave impact:** ∑ sign(imbalance)·√|imbalance|·1/(1+d) over top 20 levels

### Market Maker (Layer 4)
- 5 **ghost agents** per asset (no physical LOB orders)
- Closed-form spread: δ* = (γ+θ)σ²|I|/2 + 2Ψ + floor
- Background HJB on 30×30 grid with CFL-adaptive FTCS (dedicated thread)
- Avellaneda-Stoikov fill model: λ = 100·exp(-kδ - η|I|)
- Self-Match Prevention (SMP)

### Ruin (Layer 5)
- Premium = spread × 200 fills/time
- Claims on LOB stress (|impact|×10⁴ > 10.0) + Poisson background (0.05/yr)
- Cramér-Lundberg ruin probability with exponential claims
- Gerber-Shiu IDE: Picard iteration, 200-point grid, every 200 steps

### Topology (Layer 6)
- EWMA α=0.005 with adaptive warm-up (first 2N samples)
- Marčenko-Pastur cleaning with effective N_eff = min(2/α, t)
- Higham alternating projections (50 iterations) for nearest correlation matrix
- Prim's MST O(N²) + greedy PMFG approximation (no planarity test)
- Fiedler eigenvalue, Brandes betweenness, local clustering
- Implicit Euler Laplacian contagion diffusion (D=0.1)

### TDA (Layer 7)
- Vietoris-Rips filtration up to tetrahedra, ε_max = 2.0
- Native Z₂ boundary matrix reduction (standard persistence algorithm)
- TRI = Σ (d-b)^p / (1+b), p=2.0 — heuristic, NOT L₁ norm
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

1. **GUDHI Build Bug:** `CMakeLists.txt` option `SOVEREIGN_USE_GUDHI` does not define the `SOVEREIGN_HAS_GUDHI` preprocessor macro. The GUDHI code path in `persistence.hpp` is unreachable. Fix: add `target_compile_definitions(sovereign_lib PUBLIC SOVEREIGN_HAS_GUDHI)` inside the GUDHI conditional.

2. **TRI → θ Feedback Not Implemented:** The design target θ(t) = θ₀·exp(α·TRI(t)) coupling Layer 7 to Layer 4 is not connected. The ambiguity parameter θ remains static at 0.5.

3. **Ghost Market Makers:** MMs evaluate spreads and simulate fills internally but never place physical limit orders into the LOB. Their bankruptcy does not remove visible liquidity.

4. **PMFG Approximation:** The engine uses greedy edge enrichment without Boyer-Myrvold planarity testing. The resulting graph is not guaranteed to be genus-0 planar.

5. **Uniform Hawkes Weights:** All 10 sum-of-exp components use uniform α_m, degrading true power-law long-memory into generic multi-exponential decay.

6. **Diagonal Variance Compensator:** The Markovian fBm variance V(t) drops cross-factor covariances, introducing a small negative bias.

7. **Legacy Dead Code:** `VolterraFBM` and `FBMGenerator` in `core/random.hpp` are complete but unused. Config parameter `n_hybrid_steps` is parsed but never read.

8. **Orphaned MLMC:** `mc/mlmc.hpp` contains a complete multi-level Monte Carlo implementation (Giles 2008) but is never called by the engine.

9. **No Native Calibration:** All parameters come from `default_config.json`. No MLE/GMM calibration pipeline exists.

10. **Wasserstein Upper Bound:** The greedy O(n²) matching is an upper bound on optimal transport, not exact.

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
| `theta` | 0.5 | `MarketMakerConfig` |
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
│   │   ├── random.hpp                # Xoshiro256**, FBMGenerator, VolterraFBM (legacy)
│   │   └── state.hpp                 # AssetState, SimulationState
│   ├── price/
│   │   ├── rough_vol.hpp             # MarkovianFBM + RoughVolEngine
│   │   ├── levy_jumps.hpp            # CGMYEngine + CIR subordinator
│   │   └── regime.hpp                # RegimeEngine (5-state HMM)
│   ├── hawkes/
│   │   ├── multivariate.hpp          # HawkesEngine (N×5×10)
│   │   ├── kernels.hpp               # PowerLawKernel, ExponentialKernel, SumExpKernel
│   │   └── stability.hpp             # Dykstra spectral radius projector
│   ├── orderbook/
│   │   └── lob.hpp                   # OrderBook + LOBEngine
│   ├── market_maker/
│   │   └── robust_control.hpp        # MarketMakerEngine + HJB PDE
│   ├── ruin/
│   │   └── gerber_shiu.hpp           # RuinEngine + Cramér-Lundberg + Picard IDE
│   ├── topology/
│   │   ├── correlation.hpp           # EWMA + RMT + Higham
│   │   ├── graphs.hpp                # Prim MST + PMFG approximation
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
