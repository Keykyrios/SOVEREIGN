"""
SOVEREIGN Dashboard — Main window entry point.
Run:  python sovereign_dashboard.py [path/to/sim_state.json]
"""

import sys
import numpy as np

from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QStatusBar, QLabel
from PyQt6.QtCore import QTimer, QThread
from PyQt6.QtGui import QFont

# Import base
from dashboard import SimData, SS, ACCENT, WARN, ACCENT3, TEXT, GRID
from dashboard_tabs import (
    MarketTab, OrderBookTab, SurfaceTab,
    TopologyTab, TDATab, CrisisTab
)


class SovereignDashboard(QMainWindow):
    def __init__(self, sim_path='sim_state.json'):
        super().__init__()
        self.data = SimData(path=sim_path)

        self.setWindowTitle('SOVEREIGN — Stochastic Order-driven Volatility Engine')
        self.resize(1600, 950)
        self.setStyleSheet(SS)

        # ── Tabs ─────────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tab_market   = MarketTab(self.data)
        self.tab_orderbook= OrderBookTab(self.data)
        self.tab_surface  = SurfaceTab(self.data)
        self.tab_topology = TopologyTab(self.data)
        self.tab_tda      = TDATab(self.data)
        self.tab_crisis   = CrisisTab(self.data)

        self.tabs.addTab(self.tab_market,    '📈  Market')
        self.tabs.addTab(self.tab_orderbook, '📖  Order Book')
        self.tabs.addTab(self.tab_surface,   '🌐  3D Surfaces')
        self.tabs.addTab(self.tab_topology,  '🕸  Topology')
        self.tabs.addTab(self.tab_tda,       '⬡  TDA / Risk')
        self.tabs.addTab(self.tab_crisis,    '🚨  Crisis')

        self.setCentralWidget(self.tabs)

        # ── Status bar ───────────────────────────────────────────────────────
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.status_label = QLabel('⏳  Waiting for simulation data…')
        self.status_label.setFont(QFont('Consolas', 9))
        sb.addPermanentWidget(self.status_label)
        self.fps_label = QLabel('FPS: —')
        self.fps_label.setFont(QFont('Consolas', 9))
        sb.addWidget(self.fps_label)

        # ── Timer ────────────────────────────────────────────────────────────
        self._last_update = 0.0
        self._frame_count = 0
        self._fps = 0.0

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(150)  # 150 ms poll interval ≈ 6 FPS max

        # 3D surfaces update slower (expensive)
        self.timer_3d = QTimer()
        self.timer_3d.timeout.connect(self._tick_3d)
        self.timer_3d.start(500)

    def _tick(self):
        import time
        t0 = time.perf_counter()

        updated = self.data.poll()

        if updated or self.data.snap:
            snap = self.data.snap
            N = self.data.N()

            # Update non-3D tabs
            current = self.tabs.currentIndex()
            if   current == 0: self.tab_market.update()
            elif current == 1: self.tab_orderbook.update()
            elif current == 3: self.tab_topology.update()
            elif current == 4: self.tab_tda.update()
            elif current == 5: self.tab_crisis.update()

            # Status bar always updated
            self.tab_crisis.update()  # keeps metrics live

            step   = snap.get('step', 0) if snap else 0
            t_sim  = snap.get('t', 0.0) if snap else 0.0
            evts   = snap.get('total_events', 0) if snap else 0
            fiedler= snap.get('fiedler', 1.0) if snap else 1.0
            tri    = snap.get('tri', 0.0) if snap else 0.0

            color = WARN if fiedler < 0.1 else (ACCENT if tri < 1.0 else '#f59e0b')
            self.status_label.setText(
                f'<span style="color:{color}">■</span>'
                f'  Step {step:,} | t={t_sim:.4f} | N={N} | '
                f'Events={evts:,} | Fiedler={fiedler:.4f} | TRI={tri:.3f}'
            )

        # FPS
        import time
        now = time.perf_counter()
        self._frame_count += 1
        dt = now - self._last_update
        if dt > 1.0:
            self._fps = self._frame_count / dt
            self.fps_label.setText(f'FPS: {self._fps:.1f}')
            self._frame_count = 0
            self._last_update = now

    def _tick_3d(self):
        if self.tabs.currentIndex() == 2:
            self.tab_surface.update()
            self.tab_topology.update()





class EngineManager(QThread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.proc = None

    def run(self):
        import subprocess
        import time
        while self.running:
            self.proc = subprocess.Popen(["build\\sovereign.exe", "--assets", "10", "--steps", "3000", "--dt", "0.001"],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.proc.wait()
            time.sleep(1)

    def stop(self):
        self.running = False
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.wait()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont('Consolas', 9))

    path = sys.argv[1] if len(sys.argv) > 1 else 'sim_state.json'
    win = SovereignDashboard(sim_path=path)
    win.show()

    engine = EngineManager()
    engine.start()

    ret = app.exec()
    engine.stop()
    sys.exit(ret)


if __name__ == '__main__':
    main()
