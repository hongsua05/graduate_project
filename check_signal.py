# -*- coding: utf-8 -*-
"""
signal_check.py -- 신호 품질 점검 (표로만 출력, 그림/파일은 저장하지 않음)

두 가지 방법:
  1) 저장된 파일 분석 (기기 필요 없음)
       python signal_check.py --data mi_raw.csv
       python signal_check.py --data mini_raw_sub-1_20261020_180927.csv --out_dir out
     파일 형식: mini_session.py가 저장한 탭 구분 35열 (1~16열 = CH1~CH16, 34열 = 타임스탬프)
  2) 기기에서 바로 측정 (전극 붙이면서 확인용, 기본 30초)
       python signal_check.py --live --port COM3 --sec 30

보는 법
  - 60Hz 봉우리(mains/floor)가 작을수록 좋다. 10배 이하면 양호, 100배 이상이면 접촉/기준/환경 문제.
  - 2~40Hz가 오른쪽으로 내려가는 모양이어야 뇌파. 평평하면 잡음 바닥에 묻힌 것.
  - 채널별 표준편차가 서로 크게 다르면 접촉 불량 채널.
"""
import argparse, time
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, iirnotch, filtfilt

NAMES = ["Fp1", "F7", "T7", "P7", "Fp2", "F8", "T8", "P8",
                 "F3", "C3", "P3", "F4", "C4", "P4", "Cz", "Pz", "O1", "O2"]
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def load_file(path, ts_col=33):
    d = np.loadtxt(path, delimiter="\t")
    return d[:, 1:17].T, d[:, ts_col]


def load_live(port, sec):
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    p = BrainFlowInputParams(); p.serial_port = port
    bid = BoardIds.IRONBCI_32_BOARD
    b = BoardShim(bid, p)
    b.prepare_session(); b.start_stream(450000)
    try:
        print(f"{sec}초 측정 중... 가만히 앉아 있기")
        time.sleep(sec)
        data = b.get_board_data()
    finally:
        try: b.stop_stream()
        finally: b.release_session()
    eeg = BoardShim.get_eeg_channels(bid)[:16]
    return data[eeg, :], data[BoardShim.get_timestamp_channel(bid)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="저장된 raw 파일 경로")
    ap.add_argument("--live", action="store_true", help="기기에서 바로 측정")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--sec", type=int, default=30)
    ap.add_argument("--mains", type=float, default=60.0, help="전원 주파수 (한국 60)")
    ap.add_argument("--out_dir", default=".", help="(사용 안 함: 이제 그림을 저장하지 않아)")
    ap.add_argument("--t0", type=float, default=10.0, help="(사용 안 함)")
    ap.add_argument("--win", type=float, default=10.0, help="(사용 안 함)")
    a = ap.parse_args()
    if not a.data and not a.live:
        raise SystemExit("--data 파일경로 또는 --live 중 하나를 써줘")

    X, ts = load_live(a.port, a.sec) if a.live else load_file(a.data)
    fs = 1.0 / np.median(np.diff(ts))
    print(f"샘플 {X.shape[1]}개, fs≈{fs:.1f}Hz, 길이 {X.shape[1]/fs:.0f}초")

    f, P = welch(X, fs=fs, nperseg=int(fs * 4), axis=-1, detrend="linear")

    def peak_ratio(i, c):
        pk = P[i][np.abs(f - c) < 1].max()
        fl = np.median(P[i][(np.abs(f - c) > 3) & (np.abs(f - c) < 8)])
        return pk / fl

    tot = _trapz(P[:, (f >= 1) & (f <= 40)], f[(f >= 1) & (f <= 40)], axis=1)
    al = _trapz(P[:, (f >= 8) & (f <= 12)], f[(f >= 8) & (f <= 12)], axis=1)
    floor = np.median(P[:, (f >= 80) & (f <= 100)], axis=1) if f[-1] >= 100 else np.median(P[:, (f >= 20) & (f <= 40)], axis=1)

    # 채널별 지표 + 다른 채널 중앙값과 비교한 자동 판정 (절대 기준보다 상대 비교가 접촉 불량을 더 잘 잡는다)
    _nb, _na = iirnotch(a.mains, 30, fs)
    Xs = sosfiltfilt(butter(4, [1, 40], btype="band", fs=fs, output="sos"), filtfilt(_nb, _na, X, axis=1), axis=1)
    sd = Xs.std(axis=1)                              # 뇌파 대역(1-40Hz)만의 진폭 (60Hz 제외)
    lo = (f >= 4) & (f <= 10)
    slope = np.median(P[:, lo], axis=1) / floor        # 저주파/고주파. 뇌파면 크고, 잡음만 들어오면 1에 가깝다
    sd_med, fl_med, sl_med = np.median(sd), np.median(floor), np.median(slope)

    print(f"\n{'ch':<5}{'DC mean':>10}{'std':>9}{'mains/floor':>13}{'alpha/total':>13}{'floor PSD':>11}{'slope':>8}  판정")
    ratios = []; bad_list = []
    for i, n in enumerate(NAMES[:X.shape[0]]):
        r = peak_ratio(i, a.mains); ratios.append(r)
        flags = []
        if sd[i] > 2.5 * sd_med: flags.append("진폭 큼")
        if sd[i] < 0.4 * sd_med: flags.append("진폭 작음(확인)")
        if floor[i] > 3 * fl_med: flags.append("바닥잡음 큼")
        if slope[i] < 0.4 * sl_med: flags.append("뇌파 기울기 없음")
        elif slope[i] < 3: flags.append("뇌파 약함(잡음에 묻힘)")
        if flags: bad_list.append(n)
        print(f"{n:<5}{X[i].mean():>10.0f}{sd[i]:>9.1f}{r:>13.0f}{al[i]/tot[i]:>13.2f}{floor[i]:>11.0f}{slope[i]:>8.1f}  " + (", ".join(flags) if flags else "OK"))
    print("판정 기준: 진폭/바닥잡음은 다른 채널 중앙값 대비, slope는 3 이상이어야 뇌파가 잡음 위로 올라온 것. 전 채널이 같이 나쁘면 기준(귀)/접지 문제.")
    print(f"\n점검이 필요한 채널: {', '.join(bad_list) if bad_list else '없음'}   (다른 채널 중앙값과 비교한 기준)")
    med = float(np.median(ratios))
    print(f"\n{a.mains:.0f}Hz 봉우리/바닥 중앙값: {med:.0f}배")
    if med < 10: print("판정: 전원 잡음 작음 (양호)")
    elif med < 100: print("판정: 전원 잡음 있음. 기준/접지 접촉, 충전기 분리를 점검")
    else: print("판정: 전원 잡음이 매우 큼. 기준(귀)/접지 전극 접촉, 배터리 구동, 주변 전자기기를 점검")


if __name__ == "__main__":
    main()