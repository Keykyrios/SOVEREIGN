"""
SOVEREIGN Dashboard — Tab panels.
All data sourced exclusively from SimData which reads the C++ engine output.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSplitter, QGroupBox, QFrame
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from dashboard import SimData, mkplot, PALETTE, PANEL_BG, GRID
from dashboard import ACCENT, ACCENT2, ACCENT3, WARN, TEXT


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Market Overview
# Price paths · Volatility · Hurst exponent · Regime distribution
# ─────────────────────────────────────────────────────────────────────────────
class MarketTab(QWidget):
    def __init__(self, d: SimData):
        super().__init__(); self.d = d
        self.cp, self.cv, self.ch = {}, {}, {}

        sp = QSplitter(Qt.Orientation.Vertical)
        sp.setChildrenCollapsible(False)

        self.pp = mkplot('Price Paths  P_i(t)', 'Step', 'Price ($)')
        self.pv = mkplot('Instantaneous Volatility  σ_i(t)', 'Step', 'σ')
        self.ph = mkplot('Stochastic Hurst Exponent  H_i(t)', 'Step', 'H')
        self.pr = mkplot('Ruin Probability  Γ_i(t)', 'Asset', 'Γ', yrange=(0, 1))
        self.bar_ruin = pg.BarGraphItem(x=[], height=[], width=0.7, brush=WARN)
        self.pr.addItem(self.bar_ruin)

        for w in (self.pp, self.pv, self.ph, self.pr): sp.addWidget(w)
        sp.setSizes([300, 200, 200, 150])
        QVBoxLayout(self).addWidget(sp)

    def update(self):
        N = self.d.N()
        if not N: return
        for i in range(min(N, 20)):
            col = PALETTE[i % len(PALETTE)]
            if i not in self.cp:
                self.cp[i] = self.pp.plot(pen=pg.mkPen(col, width=1.5), name=f'A{i:02d}')
                self.cv[i] = self.pv.plot(pen=pg.mkPen(col, width=1))
                self.ch[i] = self.ph.plot(pen=pg.mkPen(col, width=1))
            for curve, field in ((self.cp[i],'price'),(self.cv[i],'vol'),(self.ch[i],'hurst')):
                ts = self.d.ts(field, asset=i)
                curve.setData(np.arange(len(ts)), ts)

        ruin = self.d.ruin_vec()
        if len(ruin):
            self.bar_ruin.setOpts(x=np.arange(N), height=ruin, width=0.7)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Order Book & Microstructure
# Real LOB depth · Hawkes intensities · MM spread & inventory
# ─────────────────────────────────────────────────────────────────────────────
class OrderBookTab(QWidget):
    def __init__(self, d: SimData):
        super().__init__(); self.d = d
        self.cs, self.ci = {}, {}

        sp = QSplitter(Qt.Orientation.Vertical)
        sp.setChildrenCollapsible(False)

        # LOB depth profile — uses real bid_vol / ask_vol arrays from engine
        self.plob = mkplot('Limit Order Book Depth — Asset 0  (real engine data)', 'Level', 'Volume')
        self.bar_bid = pg.BarGraphItem(x=[], height=[], width=0.4, brush=ACCENT3)
        self.bar_ask = pg.BarGraphItem(x=[], height=[], width=0.4, brush=WARN)
        self.plob.addItem(self.bar_bid)
        self.plob.addItem(self.bar_ask)

        # Hawkes intensities — actual λ_i^{k,d}(t) exported from engine
        self.phawk = mkplot('Hawkes Intensities  λ_i(t)  per asset', 'Asset', 'Σλ(t)')
        self.bar_hawk = pg.BarGraphItem(x=[], height=[], width=0.7, brush='#a855f7')
        self.phawk.addItem(self.bar_hawk)

        # Spread and inventory time series
        self.psp  = mkplot('MM Spread  δ_i(t)', 'Step', 'Spread')
        self.pinv = mkplot('MM Inventory  I_i(t)', 'Step', 'Inventory')

        for w in (self.plob, self.phawk, self.psp, self.pinv): sp.addWidget(w)
        sp.setSizes([250, 200, 200, 200])
        QVBoxLayout(self).addWidget(sp)

    def update(self):
        N = self.d.N()
        if not N: return

        # Real LOB depth from engine
        bids = self.d.lob_bid(0)
        asks = self.d.lob_ask(0)
        L = len(bids)
        if L:
            bid_x = -np.arange(L) - 0.7
            ask_x =  np.arange(L) + 0.3
            self.bar_bid.setOpts(x=bid_x, height=bids, width=0.4)
            self.bar_ask.setOpts(x=ask_x, height=asks, width=0.4)

        # Real Hawkes intensities summed over k,d per asset
        haw_agg = np.array([self.d.hawkes(i).sum() for i in range(N)])
        self.bar_hawk.setOpts(x=np.arange(N), height=haw_agg, width=0.7)

        # MM spread + inventory
        for i in range(min(N, 10)):
            col = PALETTE[i % len(PALETTE)]
            if i not in self.cs:
                self.cs[i] = self.psp.plot(pen=pg.mkPen(col, width=1))
                self.ci[i] = self.pinv.plot(pen=pg.mkPen(col, width=1))
            self.cs[i].setData(np.arange(len(self.d.ts('mm_spread', i))),
                               self.d.ts('mm_spread', i))
            self.ci[i].setData(np.arange(len(self.d.ts('mm_inventory', i))),
                               self.d.ts('mm_inventory', i))


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — 3D Surfaces
# Volatility surface σ(asset, time) · Correlation manifold Ω(i,j)
# Eigenvalue spectrum · Hurst surface H(asset, time)
# ─────────────────────────────────────────────────────────────────────────────
class SurfaceTab(QWidget):
    def __init__(self, d: SimData):
        super().__init__(); self.d = d
        self._vol_surf = None; self._corr_surf = None
        self._hurst_surf = None; self._eig_curve = None
        self._vol_hist = []; self._hurst_hist = []

        layout = QGridLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4,4,4,4)

        def glview(label):
            v = gl.GLViewWidget()
            from PyQt6.QtWidgets import QSizePolicy
            v.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            v.setMinimumSize(200, 200)
            v.setCameraPosition(distance=35, elevation=25, azimuth=45)
            v.setBackgroundColor(PANEL_BG)
            g = gl.GLGridItem(); g.setColor(pg.mkColor(GRID))
            g.setSize(x=15, y=15); v.addItem(g)
            gb = QGroupBox(label); vl = QVBoxLayout(gb); vl.addWidget(v)
            return gb, v

        gb1, self.gl_vol   = glview('Volatility Surface  σ(asset, time)')
        gb2, self.gl_corr  = glview('Correlation Manifold  Ω(i,j,t)')
        gb3, self.gl_hurst = glview('Hurst Exponent Surface  H(asset, time)')

        # Eigenvalue spectrum 2D
        self.p_eig = mkplot('Eigenvalue Spectrum  of  Ω(t)', 'Index', 'λ_k')
        self.c_eig = self.p_eig.plot(pen=None, symbol='o',
                                      symbolBrush=ACCENT, symbolSize=5)
        gb4 = QGroupBox('RMT Eigenvalue Spectrum')
        vl4 = QVBoxLayout(gb4); vl4.addWidget(self.p_eig)

        layout.addWidget(gb1, 0, 0)
        layout.addWidget(gb2, 0, 1)
        layout.addWidget(gb3, 1, 0)
        layout.addWidget(gb4, 1, 1)
        
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)

    @staticmethod
    def _colorize(Z: np.ndarray) -> np.ndarray:
        """Map 2D array to RGBA float32 via thermal colormap — no external deps."""
        z = (Z - Z.min()) / (np.ptp(Z) + 1e-12)
        r = np.clip(1.5 * z - 0.5, 0, 1)
        g = np.clip(2.0 * z * (1 - z) * 3, 0, 1)
        b = np.clip(1 - 2*z, 0, 1)
        a = np.full_like(z, 0.9)
        return np.stack([r, g, b, a], axis=-1).astype(np.float32)

    def _add_surf(self, glw, item, Z, xs, ys):
        cols = self._colorize(Z)
        if item is None:
            item = gl.GLSurfacePlotItem(
                x=xs.astype(np.float32),
                y=ys.astype(np.float32),
                z=Z.astype(np.float32),
                colors=cols, shader='shaded', smooth=True)
            glw.addItem(item)
        else:
            item.setData(x=xs.astype(np.float32),
                         y=ys.astype(np.float32),
                         z=Z.astype(np.float32),
                         colors=cols)
        return item

    def update(self):
        N = self.d.N()
        if N < 2: return

        vol_row   = self.d.asset_vec('vol')
        hurst_row = self.d.asset_vec('hurst')
        if not len(vol_row): return

        self._vol_hist.append(vol_row.copy())
        self._hurst_hist.append(hurst_row.copy())
        if len(self._vol_hist) > 80:   self._vol_hist.pop(0)
        if len(self._hurst_hist) > 80: self._hurst_hist.pop(0)

        T = len(self._vol_hist)
        if T < 2: return

        T_MAX = 80
        xs = np.linspace(0, 10, T_MAX, dtype=np.float32)
        ys = np.linspace(0, 10, N, dtype=np.float32)

        Zv = np.zeros((T_MAX, N), dtype=np.float32)
        Zh = np.zeros((T_MAX, N), dtype=np.float32)
        
        v_hist = np.array(self._vol_hist, dtype=np.float32) * 50
        h_hist = np.array(self._hurst_hist, dtype=np.float32) * 20
        
        pad = T_MAX - T
        Zv[pad:, :] = v_hist
        Zh[pad:, :] = h_hist
        if pad > 0:
            Zv[:pad, :] = v_hist[0, :]
            Zh[:pad, :] = h_hist[0, :]

        self._vol_surf   = self._add_surf(self.gl_vol,   self._vol_surf,   Zv, xs, ys)
        self._hurst_surf = self._add_surf(self.gl_hurst, self._hurst_surf, Zh, xs, ys)

        # Correlation manifold
        C = self.d.corr_matrix()
        xc = np.linspace(0, 10, N, dtype=np.float32)
        Zc = C.astype(np.float32) * 5
        self._corr_surf = self._add_surf(self.gl_corr, self._corr_surf, Zc, xc, xc)

        # Eigenvalue spectrum
        eigs = self.d.eigenvalues()
        if len(eigs):
            eigs_s = np.sort(eigs)[::-1]
            self.c_eig.setData(np.arange(len(eigs_s)), eigs_s)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Topology & Contagion
# Correlation heatmap · 3D network graph · Fiedler value · Betweenness
# ─────────────────────────────────────────────────────────────────────────────
class TopologyTab(QWidget):
    def __init__(self, d: SimData):
        super().__init__(); self.d = d
        self._nodes = None

        layout = QGridLayout(self); layout.setSpacing(4)
        layout.setContentsMargins(4,4,4,4)

        # Correlation heatmap — hand-built diverging colormap, no external deps
        self.img = pg.ImageView()
        pos = np.array([0.0, 0.5, 1.0])
        color = np.array([[0, 0, 255, 255], [255, 255, 255, 255], [255, 0, 0, 255]], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.img.setColorMap(cmap)
        self.img.ui.roiBtn.hide(); self.img.ui.menuBtn.hide()
        gb_img = QGroupBox('Correlation Matrix  Ω(t)  [RMT-cleaned]')
        QVBoxLayout(gb_img).addWidget(self.img)

        # 3D network
        self.glnet = gl.GLViewWidget()
        self.glnet.setCameraPosition(distance=30, elevation=20, azimuth=45)
        self.glnet.setBackgroundColor(PANEL_BG)
        g = gl.GLGridItem(); g.setColor(pg.mkColor(GRID))
        g.setSize(x=20, y=20); self.glnet.addItem(g)
        gb_net = QGroupBox('Asset Network  (node size = betweenness, color = Γ)')
        QVBoxLayout(gb_net).addWidget(self.glnet)

        # Fiedler
        self.pf = mkplot('Fiedler Value  λ₂(L(t))  — → 0 = fragmentation', 'Step', 'λ₂')
        self.cf = self.pf.plot(pen=pg.mkPen(ACCENT, width=2))
        self.pf.addLine(y=0.1, pen=pg.mkPen(WARN, style=Qt.PenStyle.DashLine))

        # Betweenness bar
        self.pb = mkplot('Betweenness Centrality  b_i(t)', 'Asset', 'b')
        self.bar_btw = pg.BarGraphItem(x=[], height=[], width=0.7, brush=ACCENT2)
        self.pb.addItem(self.bar_btw)

        layout.addWidget(gb_img, 0, 0)
        layout.addWidget(gb_net, 0, 1)
        layout.addWidget(self.pf, 1, 0)
        layout.addWidget(self.pb, 1, 1)

    def update(self):
        N = self.d.N()
        if not N: return

        C = self.d.corr_matrix()
        self.img.setImage(C.T, autoRange=False, levels=(-1, 1))

        btw  = self.d.betweenness()
        ruin = self.d.ruin_vec()

        if len(btw) == N:
            self.bar_btw.setOpts(x=np.arange(N), height=btw, width=0.7)

            # 3D network: nodes on sphere, coloured by ruin, sized by betweenness
            theta = np.linspace(0, 2*np.pi, N, endpoint=False)
            phi   = np.linspace(0.2, np.pi - 0.2, N)
            R     = 8.0
            pos = np.column_stack([
                R * np.sin(phi) * np.cos(theta),
                R * np.sin(phi) * np.sin(theta),
                R * np.cos(phi)
            ]).astype(np.float32)
            sizes  = (6 + btw * 20).astype(np.float32)
            colors = np.zeros((N, 4), np.float32)
            colors[:, 0] = ruin           # R
            colors[:, 2] = 1 - ruin       # B
            colors[:, 3] = 0.9

            if self._nodes is None:
                self._nodes = gl.GLScatterPlotItem(
                    pos=pos, size=sizes, color=colors, pxMode=True)
                self.glnet.addItem(self._nodes)
            else:
                self._nodes.setData(pos=pos, size=sizes, color=colors)

        fts = self.d.ts('fiedler')
        if len(fts): self.cf.setData(np.arange(len(fts)), fts)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — TDA / Risk Index
# TRI · W₂ · Landscape norms · Persistence diagram (from ruin/distance data)
# ─────────────────────────────────────────────────────────────────────────────
class TDATab(QWidget):
    def __init__(self, d: SimData):
        super().__init__(); self.d = d

        layout = QGridLayout(self); layout.setSpacing(4)
        layout.setContentsMargins(4,4,4,4)

        self.ptri = mkplot('Topological Risk Index  TRI(t)', 'Step', 'TRI')
        self.ctri = self.ptri.plot(pen=pg.mkPen(WARN, width=2.5))

        self.pw2  = mkplot('Wasserstein-2 Distance  W₂(PD(t), PD(t-1))', 'Step', 'W₂')
        self.cw2  = self.pw2.plot(pen=pg.mkPen(ACCENT2, width=2))

        self.pl12 = mkplot('Persistence Landscape Norms  ||λ||₁  ||λ||₂', 'Step', 'Norm')
        self.pl12.addLegend()
        self.cl1 = self.pl12.plot(pen=pg.mkPen(ACCENT,  width=1.5), name='||λ||₁')
        self.cl2 = self.pl12.plot(pen=pg.mkPen(ACCENT3, width=1.5), name='||λ||₂')

        # Persistence diagram: birth-death pairs derived from distance matrix
        # Uses the engine-exported distance matrix (Layer 6) to compute H₀ pairs
        self.ppers = mkplot('Persistence Diagram  H₀ / H₁  (from distance matrix)', 'birth', 'death')
        self.ppers.addLine(x=None, y=None,
            pen=pg.mkPen('#334455', width=1, style=Qt.PenStyle.DashLine))
        self.sc_h0 = pg.ScatterPlotItem(size=6, pen=None,
                                         brush=pg.mkBrush(ACCENT+'aa'))
        self.sc_h1 = pg.ScatterPlotItem(size=8, pen=None,
                                         brush=pg.mkBrush(WARN+'aa'))
        self.ppers.addItem(self.sc_h0)
        self.ppers.addItem(self.sc_h1)
        # Diagonal reference
        self.ppers.plot([0,2],[0,2], pen=pg.mkPen('#334455',width=1,
                                    style=Qt.PenStyle.DashLine))

        layout.addWidget(self.ptri,  0, 0)
        layout.addWidget(self.pw2,   0, 1)
        layout.addWidget(self.pl12,  1, 0)
        layout.addWidget(self.ppers, 1, 1)

    def _h0_pairs(self, dist: np.ndarray):
        """
        Compute H₀ persistence pairs from distance matrix via Union-Find.
        This is the same algorithm as the C++ engine — exact, not synthetic.
        """
        N = dist.shape[0]
        edges = sorted(
            [(dist[i,j], i, j) for i in range(N) for j in range(i+1,N)])
        parent = list(range(N))
        birth  = [0.0] * N
        pairs  = []

        def find(x):
            while parent[x] != x: parent[x]=parent[parent[x]]; x=parent[x]
            return x

        for w, i, j in edges:
            pi, pj = find(i), find(j)
            if pi != pj:
                younger = pi if birth[pi] >= birth[pj] else pj
                pairs.append((birth[younger], w))
                older = pj if younger == pi else pi
                parent[younger] = older
        return pairs

    def update(self):
        N = self.d.N()
        if not N: return

        ts_tri = self.d.ts('tri')
        ts_w2  = self.d.ts('wasserstein')
        ts_l1  = self.d.ts('l1')
        ts_l2  = self.d.ts('l2')
        xs = lambda ts: np.arange(len(ts))

        self.ctri.setData(xs(ts_tri), ts_tri)
        self.cw2.setData(xs(ts_w2),   ts_w2)
        self.cl1.setData(xs(ts_l1),   ts_l1)
        self.cl2.setData(xs(ts_l2),   ts_l2)

        # Persistence diagram from real distance matrix
        D = self.d.dist_matrix()
        if D.shape[0] > 1:
            pairs = self._h0_pairs(D)
            if pairs:
                births = np.array([p[0] for p in pairs])
                deaths = np.array([p[1] for p in pairs])
                self.sc_h0.setData(x=births, y=deaths)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 — Crisis Dashboard
# Aggregated systemic risk metrics + phase transition detection
# ─────────────────────────────────────────────────────────────────────────────
class CrisisTab(QWidget):
    def __init__(self, d: SimData):
        super().__init__(); self.d = d

        layout = QGridLayout(self); layout.setSpacing(6)
        layout.setContentsMargins(8,8,8,8)

        def metric(title):
            f = QFrame()
            f.setStyleSheet(f'background:{PANEL_BG};border:1px solid #1a2535;border-radius:6px;')
            v = QVBoxLayout(f)
            t = QLabel(title)
            t.setStyleSheet('color:#667788;font-size:10px;')
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.addWidget(t)
            val = QLabel('—')
            val.setStyleSheet(f'color:{ACCENT};font-size:22px;font-weight:bold;')
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.addWidget(val)
            return f, val

        self.m_fiedler, self.v_fiedler = metric('Fiedler  λ₂')
        self.m_tri,     self.v_tri     = metric('TRI(t)')
        self.m_ruin,    self.v_ruin    = metric('max Γ(t)')
        self.m_events,  self.v_events  = metric('Events')
        self.m_step,    self.v_step    = metric('Step')
        self.m_w2,      self.v_w2      = metric('W₂')

        for col, (m,_) in enumerate([
            (self.m_fiedler,None),(self.m_tri,None),(self.m_ruin,None),
            (self.m_events,None),(self.m_step,None),(self.m_w2,None)]):
            layout.addWidget(m, 0, col)

        # Crisis composite index — all derived from engine telemetry
        self.pc = mkplot('Crisis Composite Index  (all from engine)', 'Step', 'Index [0,1]')
        self.pc.addLegend()
        self.cc_fi  = self.pc.plot(pen=pg.mkPen(ACCENT,  width=2), name='1−Fiedler')
        self.cc_tri = self.pc.plot(pen=pg.mkPen(WARN,    width=2), name='TRI (norm)')
        self.cc_ru  = self.pc.plot(pen=pg.mkPen(ACCENT2, width=2), name='max(Γ)')
        self.pc.setYRange(0, 1)
        layout.addWidget(self.pc, 1, 0, 1, 6)

        self.alert = QLabel('● WAITING FOR ENGINE DATA')
        self.alert.setStyleSheet(f'color:#667788;font-size:15px;font-weight:bold;')
        self.alert.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.alert, 2, 0, 1, 6)

    def update(self):
        d = self.d
        if not d.snap: return

        fiedler = d.scalar('fiedler', 1.0)
        tri     = d.scalar('tri', 0.0)
        w2      = d.scalar('wasserstein', 0.0)
        events  = int(d.scalar('total_events', 0))
        step    = int(d.scalar('step', 0))
        ruin    = d.ruin_vec()
        max_ru  = float(ruin.max()) if len(ruin) else 0.0

        def colored(v, label, lo_warn, hi_warn, fmt='.4f'):
            c = WARN if v > hi_warn else (ACCENT2 if v > lo_warn else ACCENT)
            return v, c

        fi, fi_c = colored(fiedler, 'fiedler', 0.5, 0.8)
        # Fiedler: LOW = crisis → invert warning direction
        fi_c = WARN if fiedler < 0.1 else (ACCENT2 if fiedler < 0.4 else ACCENT)
        tr, tr_c = colored(tri, 'tri', 1.0, 5.0)
        ru, ru_c = colored(max_ru, 'ruin', 0.3, 0.7)

        def set_val(widget, text, color):
            widget.setText(text)
            widget.setStyleSheet(f'color:{color};font-size:22px;font-weight:bold;')

        set_val(self.v_fiedler, f'{fiedler:.4f}', fi_c)
        set_val(self.v_tri,     f'{tri:.3f}',     tr_c)
        set_val(self.v_ruin,    f'{max_ru:.4f}',  ru_c)
        self.v_events.setText(f'{events:,}')
        self.v_step.setText(str(step))
        self.v_w2.setText(f'{w2:.4f}')

        # Time series
        fts  = d.ts('fiedler')
        tts  = d.ts('tri')
        rtss = [d.ts('ruin', i) for i in range(min(d.N(), 1))]

        def norm_ts(a): return a / (a.max() + 1e-12)

        xs = np.arange(len(fts))
        self.cc_fi.setData(xs, np.clip(1 - fts, 0, 1))
        xs_t = np.arange(len(tts))
        self.cc_tri.setData(xs_t, norm_ts(np.clip(tts, 0, None)))
        if rtss and len(rtss[0]):
            self.cc_ru.setData(np.arange(len(rtss[0])), rtss[0])

        # Endogenous crisis score — purely from engine data
        score = (1 - fiedler) * 0.35 + min(max_ru, 1) * 0.4 + min(tri / 10, 1) * 0.25
        if score > 0.65:
            self.alert.setText('⚠  ENDOGENOUS PHASE TRANSITION DETECTED')
            self.alert.setStyleSheet(f'color:{WARN};font-size:15px;font-weight:bold;')
        elif score > 0.35:
            self.alert.setText('⚡  STRESSED — Liquidity Crisis Forming')
            self.alert.setStyleSheet(f'color:{ACCENT2};font-size:15px;font-weight:bold;')
        elif d.snap:
            self.alert.setText('●  STABLE — Monitoring Engine Output')
            self.alert.setStyleSheet(f'color:{ACCENT3};font-size:15px;font-weight:bold;')
