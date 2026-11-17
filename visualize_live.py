"""기기에서 들어오는 신호 파형을 그대로 실시간으로 보여준다 (저장 안 함, 파일 안 만듦).
실행:  python visualize.py            (기기, COM3)
       python visualize.py --synthetic (기기 없이 가짜 신호로 테스트)
옵션:  --sec 5 (화면에 보이는 초)  --ch 16 (채널 수)  --filter (1-50Hz+60Hz노치로 보기)
창을 닫거나 Ctrl+C 로 종료. decoder_server.py 와 동시에 못 켠다 (COM3 하나를 같이 못 씀).
"""
import argparse, time, sys
import numpy as np
import matplotlib
try: matplotlib.use("TkAgg")
except Exception: pass
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="COM3"); ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--sec", type=float, default=5.0); ap.add_argument("--ch", type=int, default=16)
ap.add_argument("--filter", action="store_true"); ap.add_argument("--scale", type=float, default=0, help="채널 간격(uV). 0=자동")
a = ap.parse_args()
NAMES = ["Fp1","F7","T7","P7","Fp2","F8","T8","P8","F3","C3","P3","F4","C4","P4","Cz","Pz"]

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
params = BrainFlowInputParams()
if a.synthetic: bid = BoardIds.SYNTHETIC_BOARD
else:
    try:
        import config; bid = getattr(config, "BOARD_ID", BoardIds.IRONBCI_32_BOARD); params.serial_port = getattr(config, "SERIAL_PORT", a.port)
    except Exception:
        bid = BoardIds.IRONBCI_32_BOARD; params.serial_port = a.port
BoardShim.disable_board_logger()
board = BoardShim(bid, params); board.prepare_session(); board.start_stream(450000)
chs = list(BoardShim.get_eeg_channels(bid))[:a.ch]; ts_ch = BoardShim.get_timestamp_channel(bid)
time.sleep(2.0)
d = board.get_current_board_data(1500); ts = d[ts_ch]
fs = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 10 and ts[-1] > ts[0] else BoardShim.get_sampling_rate(bid)
print(f"샘플링 속도 약 {fs:.0f}Hz, 채널 {len(chs)}개. 창을 닫으면 종료.")
N = int(a.sec * fs)

if a.filter:
    from scipy.signal import iirnotch, butter, sosfiltfilt, filtfilt
    nb, na = iirnotch(60, 30, fs); sos = butter(4, [1, min(50, fs/2-1)], btype="band", fs=fs, output="sos")

fig, ax = plt.subplots(figsize=(12, 8)); fig.canvas.manager.set_window_title("signal viewer")
lines = [ax.plot([], [], lw=0.7)[0] for _ in chs]
ax.set_xlim(-a.sec, 0); ax.set_xlabel("time (s)"); ax.set_yticks([]); t = np.linspace(-a.sec, 0, N)
for i in range(len(chs)): ax.text(-a.sec, 0, NAMES[i] if i < len(NAMES) else f"ch{i+1}", fontsize=8, va="center", ha="right")
txt = ax.text(0.99, 0.99, "", transform=ax.transAxes, ha="right", va="top", fontsize=9)
buf = np.zeros((len(chs), N)); n_got = 0; gap = None

try:
    while plt.fignum_exists(fig.number):
        data = board.get_board_data()
        if data.shape[1] > 0:
            x = data[chs]; k = min(x.shape[1], N); n_got = min(N, n_got + k)
            buf = np.roll(buf, -k, axis=1); buf[:, -k:] = x[:, -k:]
        v = buf.copy()
        if a.filter and n_got > 200:
            v = filtfilt(nb, na, v, axis=1); v = sosfiltfilt(sos, v, axis=1)
        v = v - np.median(v, axis=1, keepdims=True)               # 직류 성분만 빼고 모양은 그대로
        sc = a.scale if a.scale > 0 else max(3 * np.percentile(np.abs(v[:, -min(N, max(n_got, 1)):]), 95), 1.0)
        for i, ln in enumerate(lines):
            ln.set_data(t, v[i] + (len(chs) - 1 - i) * sc * 2)
        ax.set_ylim(-sc * 2, len(chs) * sc * 2)
        for i, tx in enumerate([c for c in ax.texts if c is not txt]): tx.set_position((-a.sec, (len(chs) - 1 - i) * sc * 2))
        txt.set_text(f"spacing {sc*2:.0f} uV / ch   {'filtered 1-50Hz' if a.filter else 'raw (DC removed)'}")
        plt.pause(0.05)
except KeyboardInterrupt: pass
finally:
    try: board.stop_stream(); board.release_session()
    except Exception: pass