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

        self.tabs.addTab(self.tab_market,    ' Market')
        self.tabs.addTab(self.tab_orderbook, '  Order Book')
        self.tabs.addTab(self.tab_surface,   '  3D Surfaces')
        self.tabs.addTab(self.tab_topology,  '  Topology')
        self.tabs.addTab(self.tab_tda,       '  TDA / Risk')
        self.tabs.addTab(self.tab_crisis,    '  Crisis')

        self.setCentralWidget(self.tabs)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._last_rendered_tab = -1

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
        self.timer.start(16)  # 16 ms poll interval ≈ 60 FPS max

        # 3D surfaces update slightly slower (expensive, but much faster than before)
        self.timer_3d = QTimer()
        self.timer_3d.timeout.connect(self._tick_3d)
        self.timer_3d.start(33)  # ~30 FPS for 3D surfaces
        self._new_data_arrived = False  # Gate for 3D updates

    def closeEvent(self, event):
        """Cleanup data connection on window close."""
        self.data.close()
        event.accept()

    def _on_tab_changed(self, index):
        self._last_rendered_tab = -1
        self._new_data_arrived = True

    def _tick(self):
        import time
        t0 = time.perf_counter()

        updated = self.data.poll()
        current = self.tabs.currentIndex()

        if updated or self._last_rendered_tab != current:
            self._last_rendered_tab = current
            if updated:
                self._new_data_arrived = True  # Signal for 3D timer

            if self.data.snap:
                snap = self.data.snap
                N = self.data.N()

                # Update ONLY the currently visible tab
                if   current == 0: self.tab_market.update()
                elif current == 1: self.tab_orderbook.update()
                elif current == 3: self.tab_topology.update()
                elif current == 4: self.tab_tda.update()
                elif current == 5: self.tab_crisis.update()

                # Status bar always updated with new data
                if updated:
                    self.tab_crisis.update()

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
        # ONLY update 3D views when NEW data arrived from engine
        if self._new_data_arrived and self.tabs.currentIndex() == 2:
            self.tab_surface.update()
            self.tab_topology.update()
            self._new_data_arrived = False





class EngineManager(QThread):
    MAX_RETRIES = 5  # Stop spamming after 5 consecutive failures

    def __init__(self):
        super().__init__()
        self.running = True
        self.proc = None

    def run(self):
        import subprocess
        import time
        import os
        import platform

        base_dir = os.path.dirname(os.path.abspath(__file__))
        cwd_path = os.path.join(base_dir, "build")
        exe_name = "sovereign.exe" if platform.system() == "Windows" else "sovereign"
        exe_path = os.path.join(cwd_path, exe_name)

        if not os.path.exists(exe_path):
            print(f"Engine binary not found: {exe_path}")
            print("Build the engine first: cmake --build build --config Release")
            return

        failures = 0
        while self.running:
            try:
                self.proc = subprocess.Popen(
                    [exe_path, "--assets", "50", "--steps", "500000", "--dt", "0.00002"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    cwd=cwd_path
                )
                self.proc.wait()
                failures = 0  # Reset on successful run
            except Exception as e:
                failures += 1
                print(f"Engine launch failed ({failures}/{self.MAX_RETRIES}): {e}")
                if failures >= self.MAX_RETRIES:
                    print("Max retries reached. Stopping engine manager.")
                    return
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

    path = sys.argv[1] if len(sys.argv) > 1 else 'build/sim_state.json'
    win = SovereignDashboard(sim_path=path)
    win.show()

    # Auto-start engine in background
    engine = EngineManager()
    engine.start()

    ret = app.exec()
    engine.stop()
    sys.exit(ret)


if __name__ == '__main__':
    main()
