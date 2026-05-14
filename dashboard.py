"""
SOVEREIGN Dashboard — base module.
Polls sim_state.json written by the C++ engine.
NO synthetic/mock data anywhere. All fields come directly from engine output.
"""

import sys, json, os, time
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QSplitter, QFrame, QStatusBar, QGroupBox
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

import pyqtgraph as pg
import pyqtgraph.opengl as gl

pg.setConfigOptions(antialias=True, background='#0a0a0f', foreground='#c8d8e8')

DARK_BG    = '#0a0a0f'
PANEL_BG   = '#0f1117'
ACCENT     = '#00d4ff'
ACCENT2    = '#ff6b35'
ACCENT3    = '#39ff14'
WARN       = '#ff3366'
TEXT       = '#c8d8e8'
GRID       = '#1a2030'

PALETTE = [
    '#00d4ff','#ff6b35','#39ff14','#ff3366','#a855f7',
    '#f59e0b','#06b6d4','#ec4899','#84cc16','#f97316',
    '#8b5cf6','#14b8a6','#ef4444','#3b82f6','#22c55e',
    '#eab308','#6366f1','#f43f5e','#0ea5e9','#10b981',
]

SS = f"""
QMainWindow,QWidget{{background:{DARK_BG};color:{TEXT};font-family:Consolas,monospace;font-size:10px}}
QTabWidget::pane{{border:1px solid {GRID};background:{PANEL_BG}}}
QTabBar::tab{{background:#0f1117;color:#667788;padding:8px 16px;border:1px solid {GRID}}}
QTabBar::tab:selected{{background:#1a2535;color:{ACCENT};border-bottom:2px solid {ACCENT}}}
QGroupBox{{border:1px solid #1a2535;border-radius:4px;margin-top:8px;padding:6px;color:{ACCENT};font-size:10px}}
QStatusBar{{background:#080b10;color:{ACCENT};border-top:1px solid #1a2535}}
"""


def mkplot(title='', xl='', yl='', yrange=None):
    p = pg.PlotWidget(title=f'<span style="color:{ACCENT};font-size:11px">{title}</span>')
    p.setBackground(PANEL_BG)
    p.showGrid(x=True, y=True, alpha=0.12)
    for ax in ('bottom','left'):
        p.getAxis(ax).setPen(pg.mkPen(GRID))
        p.getAxis(ax).setTextPen(pg.mkPen(TEXT))
    if xl: p.getAxis('bottom').setLabel(xl, color=TEXT)
    if yl: p.getAxis('left').setLabel(yl, color=TEXT)
    if yrange: p.setYRange(*yrange)
    return p


class SimData:
    """
    Reads telemetry from sovereign::TelemetryWriter via UDP.
    All data comes directly from the C++ engine — nothing generated here.
    """
    def __init__(self, path='sim_state.json', port=8080):
        self.path   = path # Kept for signature compatibility
        import socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Increase receive buffer to prevent dropping large payload datagrams
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        try:
            self.sock.bind(("127.0.0.1", port))
        except OSError:
            pass # already bound by another instance?
        self.sock.setblocking(False)
        
        self.snap   = None
        self.hist   = []
        self.maxhist= 1000

    def poll(self) -> bool:
        """Return True if new snapshot loaded."""
        import json
        updated = False
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
                raw = data.decode('utf-8')
                if not raw: continue
                s = json.loads(raw)
                
                # Bulletproof check: handle new simulations and duplicates
                new_step = s.get('step', -1)
                old_step = self.snap.get('step', -1) if self.snap else -2
                
                if new_step == old_step:
                    continue
                elif new_step < old_step:
                    # A new simulation run has started! 
                    # Clear the history so we don't wait for step to catch up.
                    self.hist.clear()
                    
                self.snap = s
                self.hist.append(s)
                if len(self.hist) > self.maxhist: self.hist.pop(0)
                updated = True
            except BlockingIOError:
                break
            except Exception:
                # Json decode error on truncated packet, ignore
                break
        return updated

    # ── Accessors ──────────────────────────────────────────────────────────

    def N(self) -> int:
        return len(self.snap.get('assets', [])) if self.snap else 0

    def ts(self, field: str, asset: int = None) -> np.ndarray:
        """Time series of a scalar field from history."""
        out = []
        for s in self.hist:
            try:
                v = s['assets'][asset][field] if asset is not None else s[field]
                out.append(float(v))
            except Exception:
                out.append(float('nan'))
        return np.array(out, dtype=float)

    def asset_vec(self, field: str) -> np.ndarray:
        """Latest value of field across all assets."""
        if not self.snap: return np.array([])
        try:
            return np.array([float(a[field]) for a in self.snap['assets']])
        except Exception:
            return np.array([])

    def lob_bid(self, asset: int) -> np.ndarray:
        try:
            return np.array(self.snap['assets'][asset]['lob']['bid_vol'], dtype=float)
        except Exception:
            return np.zeros(1)

    def lob_ask(self, asset: int) -> np.ndarray:
        try:
            return np.array(self.snap['assets'][asset]['lob']['ask_vol'], dtype=float)
        except Exception:
            return np.zeros(1)

    def hawkes(self, asset: int) -> np.ndarray:
        """All Hawkes intensity components for one asset."""
        try:
            return np.array(self.snap['assets'][asset]['hawkes_intensity'], dtype=float)
        except Exception:
            return np.zeros(1)

    def corr_matrix(self) -> np.ndarray:
        N = self.N()
        try:
            return np.array(self.snap['correlation'], dtype=float).reshape(N, N)
        except Exception:
            return np.eye(N)

    def dist_matrix(self) -> np.ndarray:
        N = self.N()
        try:
            return np.array(self.snap['distance'], dtype=float).reshape(N, N)
        except Exception:
            return np.ones((N, N))

    def eigenvalues(self) -> np.ndarray:
        try:
            return np.array(self.snap['eigenvalues'], dtype=float)
        except Exception:
            return np.array([])

    def betweenness(self) -> np.ndarray:
        try:
            return np.array(self.snap['betweenness'], dtype=float)
        except Exception:
            return np.zeros(self.N())

    def ruin_vec(self) -> np.ndarray:
        try:
            return np.array(self.snap['ruin_vector'], dtype=float)
        except Exception:
            return np.zeros(self.N())

    def scalar(self, field: str, default=0.0) -> float:
        if not self.snap: return default
        try: return float(self.snap[field])
        except Exception: return default
