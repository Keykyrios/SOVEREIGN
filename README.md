<div align="center">
  <h1>SOVEREIGN</h1>
  <h3>High-Frequency Microstructure, Actuarial Contagion, and Topological Data Analysis</h3>
  <br />
  <a href="https://drive.google.com/file/d/1KKBpKEfQiJph58u0bLJ2V7AA1c_fbWbE/view?usp=sharing"><strong>Read the Official Whitepaper (Mathematical Specification)</strong></a>
  <br /><br />
  <img src="https://img.shields.io/badge/C%2B%2B-20-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="C++20" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Qt-PyQt6-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="PyQt6" />
  <img src="https://img.shields.io/badge/OpenGL-3D_Telemetry-FFFFFF?style=for-the-badge&logo=opengl&logoColor=black" alt="OpenGL" />
  <img src="https://img.shields.io/badge/Math-Eigen3-red?style=for-the-badge" alt="Eigen3" />
  <img src="https://img.shields.io/badge/Compiler-GCC%2FMinGW64-orange?style=for-the-badge&logo=gnu&logoColor=white" alt="GCC" />
</div>

<br />

---

## Overview

**SOVEREIGN** is a hyper-performance, multi-asset market microstructure simulation engine written in **C++20** with an asynchronous **OpenGL PyQtGraph** telemetry dashboard. 

Unlike standard monte-carlo diffusions that rely on exogenous shocks to trigger crashes, SOVEREIGN is an **endogenous thermodynamic closed-loop system**. It mathematically guarantees spontaneous flash crashes by dynamically mapping the algebraic topology (shape) of the market directly to the risk-aversion of automated liquidity providers.

The engine accurately models high-frequency algorithmic trading physics, executing millions of events per second bypassing standard L1 cache bottlenecks via SIMD and cache-coherent ring buffers.

---

## The 7-Layer Feedback Architecture

The simulation is built on an interconnected 7-layer pipeline where topological invariants feedback directly into the Limit Order Book via a Robust HJB Control PDE.

```mermaid
graph TD
    sublayer_1[Layer 1: Stochastic Vol \& Jumps]
    sublayer_2[Layer 2: Hawkes Order Flow]
    sublayer_3[Layer 3: Limit Order Book]
    sublayer_4[Layer 4: Robust Control]
    sublayer_5[Layer 5: Gerber-Shiu Ruin]
    sublayer_6[Layer 6: Spectral Contagion]
    sublayer_7[Layer 7: Persistent Homology]

    %% Dependencies
    sublayer_1 -->|Volterra fBm & CGMY Jumps| sublayer_3
    sublayer_2 -->|Poisson Mutually-Exciting Orders| sublayer_3
    
    sublayer_3 -->|Tick Data| sublayer_6
    sublayer_3 -->|Adverse Selection Claims| sublayer_5
    
    sublayer_5 -->|Bankruptcy Vacuum| sublayer_2
    
    sublayer_6 -->|Marchenko-Pastur Metric Space| sublayer_7
    
    sublayer_7 -->|Topological Risk Index| sublayer_4
    
    sublayer_4 -->|Knightian Ambiguity Spreads| sublayer_3
    
    classDef core fill:#0b192c,stroke:#3b82f6,stroke-width:2px,color:#fff;
    class sublayer_1,sublayer_2,sublayer_3,sublayer_4,sublayer_5,sublayer_6,sublayer_7 core;
```

### 1. Rough Bergomi & CGMY Jumps (Layer 1)
Discards standard Itô diffusions. The volatility surface is driven by a fractional Volterra integral computed via the Bennedsen Hybrid Scheme ($O(N)$ with ring buffers). Heavy-tailed microstructure discontinuities are generated using rejection-sampled CGMY tempered stable Pareto distributions. The Box-Muller transform is completely bypassed using the hardware-accelerated **Ziggurat Algorithm** for normal distributions.

### 2. Multivariate Hawkes Processes (Layer 2)
Order flow is not constant; it is heavily autocorrelated. SOVEREIGN simulates 6 distinct event types using a mutually-exciting $6 \times 6$ point process matrix. The engine utilizes an $O(N)$ recursive **Ogata Thinning algorithm**, managing sub-millisecond concurrency by pushing events into an `std::priority_queue` EventClock.

### 3. Limit Order Book (Layer 3)
The absolute ground-truth state. To avoid L1 cache misses associated with `std::map`, the book uses discrete integer price grids mapped directly to `Eigen::VectorXd` contiguous arrays. Massive market orders execute using an implicit propagator to enforce the empirical **Square-Root Law of Price Impact**.

### 4. Market Maker Robust Control (Layer 4)
Market Makers solve a Robust Hamilton-Jacobi-Bellman (HJB) Equation under **Knightian Ambiguity**. They do not trust the reference probability measure, penalizing models via Kullback-Leibler Relative Entropy. 

### 5. Actuarial Ruin Analysis (Layer 5)
A continuous-time Cramér-Lundberg risk process monitors Market Maker bankruptcy. Using a high-performance Picard Iteration finite-difference scheme over a 200-element grid, the engine computes the **Gerber-Shiu** expected discounted penalty function. Low-priced penny stocks mathematically violate the Net Profit Condition and inevitably collapse.

### 6. Spectral Contagion \& Random Matrix Theory (Layer 6)
An $O(N^2)$ recursive EWMA covariance state updates instantly on each tick using AVX2 SIMD fused-multiply-adds. The matrix is cleaned using **Marchenko-Pastur** limits (preserving the trace) and mapped to an ultrametric distance matrix. The **Fiedler Eigenvalue** of the Graph Laplacian signals absolute systemic contagion.

### 7. Persistent Homology \& TDA (Layer 7)
The engine constructs a **Vietoris-Rips Simplicial Complex** and executes Gaussian column reduction over $\mathbb{Z}_2$ boundary matrices using the Elder Rule. The $L_1$ Persistent Landscape explicitly maps the Betti-1 homology loops (market "holes") to a single Topological Risk Index (TRI).

**The Closure:** The TRI is fed directly back into Layer 4. The topology terrifies the Market Makers $\to$ Spreads Widen $\to$ Liquidity Collapses $\to$ Hawkes Avalanches Destroy the Order Book. 

---

## Hardware & Software Stack
*   **Math Engine**: C++20, GCC/MinGW64, AVX2 SIMD hardware instructions.
*   **Linear Algebra**: `Eigen3` for vectorized matrix decomposition.
*   **RNG**: Cache-coherent `Xoshiro256**` operating at 1.5 billion ops/sec.
*   **Telemetry**: Asynchronous IPC via mapped JSON buffers.
*   **Dashboard**: Python 3.10+, `PyQt6`, `QThread` orchestration, `pyqtgraph` OpenGL 3D surface rendering.

---

## Build & Run Instructions

### Prerequisites (Windows)
1. Install [MSYS2](https://www.msys2.org/).
2. Open MSYS2 UCRT64 terminal and install the build chain:
   ```bash
   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make mingw-w64-ucrt-x86_64-eigen3 mingw-w64-ucrt-x86_64-nlohmann-json
   ```
3. Install Python requirements:
   ```bash
   pip install PyQt6 pyqtgraph numpy
   ```

### Execution
Simply run the Python orchestration script, which will automatically spawn a background `QThread` to manage the C++ binary lifecycle:
```bash
python sovereign_dashboard.py
```
> **Note**: Due to the severe $O(N^2)$ algebraic topology and recursive correlation matrix evaluations, memory pressure scales aggressively with the asset universe size. Limit to `-j1` parallel threads on constrained hardware.

---
*Developed by Keykyrios*
