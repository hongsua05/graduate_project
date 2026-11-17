# -*- coding: utf-8 -*-
"""
alpha_test.py  --  눈 뜨기/감기 알파파 테스트 (기기가 진짜 뇌파를 잡는지 확인)

원리
  눈을 감으면 머리 뒤쪽(P3/P4/Pz 근처)에서 8~12Hz(알파파)가 눈 뜰 때보다 크게 커진다.
  이게 보이면 기기와 전극이 실제 뇌파를 잡고 있다는 뜻이다. 안 보이면 접촉, 기준(귀) 전극, 전극 위치를 의심한다.

하는 일 (약 2분)
  1) 눈 뜨기 20초 -> 눈 감기 20초 -> 눈 뜨기 20초 -> 눈 감기 20초 (소리가 나면 바꾸기)
  2) 구간마다 처음 3초는 버리고(전환 직후 잡음), 채널별 알파(8~12Hz) 파워를 구함
  3) 채널별 "눈 감음 / 눈 뜸" 비율을 표로 보여줌 (보통 뒤쪽 채널에서 1.5배 이상). 그림/파일은 저장하지 않음

사용 (cmd, 가상환경 안에서)
  python alpha_test.py --port COM3
  python alpha_test.py --port COM3 --no_car          # 공통 평균 제거를 끔
  python alpha_test.py --port COM3 --n_ch 18          # O1/O2를 CH17, CH18에 꽂았을 때 (O1/O2 포함 18채널)
  python alpha_test.py --synthetic                  # 기기 없이 동작만 확인 (결과 의미 없음)

주의
  - 편하게 앉아서 눈을 감을 때 얼굴과 턱에 힘을 빼고 가만히 있기. 눈을 뜰 때는 한 점을 바라보기(눈 깜빡임 최소화).
  - 노트북은 배터리로, 보드는 보조배터리로 (평소 측정과 같은 조건)
"""
import argparse, os, sys, time
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

CHANNEL_NAMES = ["Fp1", "F7", "T7", "P7", "Fp2", "F8", "T8", "P8",
                 "F3", "C3", "P3", "F4", "C4", "P4", "Cz", "Pz", "O1", "O2"]
POSTERIOR = ["P3", "P4", "Pz", "P7", "P8", "O1", "O2"]       # 알파가 눈 감을 때 가장 잘 커지는 뒤쪽 채널
ALPHA = (8.0, 12.0)
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")   # numpy 1.x / 2.x 둘 다 지원


def beep(n=1):
    try:
        import winsound
        for _ in range(n):
            winsound.Beep(1000, 250); time.sleep(0.1)
    except Exception:
        print("\a" * n, end="", flush=True)


def countdown(sec, text):
    for s in range(sec, 0, -1):
        print(f"\r{text} ... {s}초 ", end="", flush=True)
        time.sleep(1)
    print("\r" + " " * 60 + "\r", end="")


def band_power(x, fs, lo, hi):
    f, p = welch(x, fs=fs, nperseg=int(fs * 2), axis=-1, detrend="constant")
    m = (f >= lo) & (f <= hi)
    return _trapz(p[..., m], f[m], axis=-1), f, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--sec", type=int, default=20, help="구간 하나의 길이(초)")
    ap.add_argument("--repeats", type=int, default=2, help="(뜸, 감음) 반복 횟수")
    ap.add_argument("--skip", type=float, default=3.0, help="구간 앞쪽에서 버릴 초")
    ap.add_argument("--no_car", action="store_true")
    ap.add_argument("--n_ch", type=int, default=18)
    ap.add_argument("--out_dir", default=".", help="(사용 안 함: 이제 파일을 저장하지 않아)")
    ap.add_argument("--synthetic", action="store_true", help="기기 없이 테스트 (BrainFlow 가짜 보드)")
    a = ap.parse_args()

    params = BrainFlowInputParams()
    if a.synthetic:
        board_id = BoardIds.SYNTHETIC_BOARD
    else:
        board_id = BoardIds.IRONBCI_32_BOARD
        params.serial_port = a.port
    board = BoardShim(board_id, params)

    plan = []
    for _ in range(a.repeats):
        plan += [("open", a.sec), ("closed", a.sec)]
    segs = []   # (label, t_start, t_end)

    print("보드 연결 중...", a.port if not a.synthetic else "(synthetic)")
    board.prepare_session()
    board.start_stream(450000)
    try:
        eeg_ch = BoardShim.get_eeg_channels(board_id)[:a.n_ch]
        ts_ch = BoardShim.get_timestamp_channel(board_id)
        print("연결됨. 편하게 앉아서 아래 안내를 따라줘. (소리가 나면 눈 상태를 바꿔)")
        countdown(8, "곧 시작해 -- 먼저 눈을 뜨고 한 점을 바라봐")
        for label, sec in plan:
            if label == "open":
                beep(2); msg = "눈 뜨고 한 점 바라보기"
            else:
                beep(1); msg = "눈 감기 (얼굴/턱 힘 빼기)"
            t0 = time.time()
            for s in range(sec, 0, -1):
                print(f"\r[{msg}] 남은 시간 {s}초 ", end="", flush=True)
                time.sleep(1)
            segs.append((label, t0, time.time()))
        beep(3)
        print("\r" + " " * 70 + "\r끝! 분석 중...")
        data = board.get_board_data()
    finally:
        try:
            board.stop_stream()
        finally:
            board.release_session()

    ts = data[ts_ch]
    fs = 1.0 / np.median(np.diff(ts))
    X = data[eeg_ch, :].astype(np.float64)
    names = CHANNEL_NAMES[:len(eeg_ch)]
    print(f"샘플 {X.shape[1]}개, 측정된 샘플링 속도 ≈ {fs:.1f} Hz, 채널 {len(eeg_ch)}개")

    sos = butter(4, [1, 40], btype="band", fs=fs, output="sos")
    Xf = sosfiltfilt(sos, X, axis=1)
    if not a.no_car:
        Xf = Xf - Xf.mean(axis=0, keepdims=True)

    pw = {"open": [], "closed": []}
    psd = {"open": [], "closed": []}
    for label, t0, t1 in segs:
        m = (ts >= t0 + a.skip) & (ts <= t1)
        if m.sum() < fs * 5:
            print(f"경고: {label} 구간 데이터가 너무 짧아 ({m.sum()} 샘플)"); continue
        seg = Xf[:, m]
        bp, f, p = band_power(seg, fs, *ALPHA)
        pw[label].append(bp); psd[label].append(p)
    if not pw["open"] or not pw["closed"]:
        raise SystemExit("분석할 구간이 부족해. 다시 해봐.")

    open_p = np.mean(pw["open"], axis=0); closed_p = np.mean(pw["closed"], axis=0)
    ratio = closed_p / np.maximum(open_p, 1e-12)
    # 반복 사이 일관성: 두 번 모두 감음>뜸 인지
    n_pair = min(len(pw["open"]), len(pw["closed"]))
    consistent = np.zeros(len(names), int)
    for i in range(n_pair):
        consistent += (pw["closed"][i] > pw["open"][i]).astype(int)

    print("\n" + "=" * 62)
    print(f"알파(8-12Hz) 파워: 눈 감음 / 눈 뜸  (CAR {'off' if a.no_car else 'on'})")
    print("=" * 62)
    print(f"{'채널':<6}{'뜸':>12}{'감음':>12}{'비율':>8}   {'반복일치':<8}")
    for i, n in enumerate(names):
        star = "  <== 뒤쪽" if n in POSTERIOR else ""
        print(f"{n:<6}{open_p[i]:>12.2f}{closed_p[i]:>12.2f}{ratio[i]:>8.2f}   {consistent[i]}/{n_pair}{star}")

    post_idx = [i for i, n in enumerate(names) if n in POSTERIOR]
    post_med = float(np.median(ratio[post_idx])) if post_idx else float("nan")
    post_ok = int(sum(ratio[i] >= 1.5 for i in post_idx))
    print(f"\n뒤쪽 채널({', '.join(n for n in POSTERIOR if n in names)}) 비율 중앙값: {post_med:.2f}  /  1.5배 이상인 채널 {post_ok}/{len(post_idx)}개")
    if post_med >= 1.5 and post_ok >= 2:
        verdict = "알파가 눈 감을 때 뚜렷하게 커졌어. 기기와 전극이 진짜 뇌파를 잡고 있어."
    elif post_med >= 1.2:
        verdict = "알파가 조금 커졌어. 신호는 있는데 약해. 접촉/기준 전극을 보강하고 다시 해봐."
    else:
        verdict = "알파 증가가 안 보여. 접촉, 귀 기준 전극, 전극 위치를 점검하고 다시 해봐. (눈을 안 감았거나 긴장했을 가능성도 있어)"
    print("판정:", verdict)


if __name__ == "__main__":
    main()