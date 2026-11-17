# -*- coding: utf-8 -*-
"""
bci_data.py -- 데이터 읽기/전처리 공통 모듈 (physionet_pretrain.py, finetune_eval.py 가 같이 쓴다)
TensorFlow 없이 동작한다 (numpy/scipy/pywt/mne).

핵심 약속 (사전학습과 미세조정이 반드시 같아야 한다)
  - 채널 16개, 순서 고정: Fp1,F7,T7,P7,Fp2,F8,T8,P8,F3,C3,P3,F4,C4,P4,Cz,Pz  (= 네 CH1~CH16, O1/O2 제외)
  - 샘플링 256Hz (PhysioNet 160Hz, IronBCI 512Hz 모두 256Hz로 맞춘다)
  - 트라이얼 구간: 큐 이후 0.5~3.5초 (3초 = 768샘플)
  - 트라이얼마다: 평균 제거 -> (옵션)CAR -> 1~50Hz 밴드패스 -> 웨이블릿(db4 L4) -> 256샘플 윈도우(step 128)
    (밴드패스/웨이블릿/윈도우는 언니들 train_mixed_5fold.py 와 똑같다)
"""
import os, re, glob
from fractions import Fraction
import numpy as np
import pywt
from scipy.signal import butter, lfilter, sosfiltfilt, iirnotch, filtfilt, resample_poly

FS = 256
WIN, STEP = 256, 128
SEG = (0.5, 3.5)
CH16 = ["Fp1", "F7", "T7", "P7", "Fp2", "F8", "T8", "P8",
        "F3", "C3", "P3", "F4", "C4", "P4", "Cz", "Pz"]


# ---------------------------------------------------------------- 신호 처리
def nearest_fs(fs_est):
    cands = [100, 128, 160, 200, 250, 256, 500, 512, 1000, 1024]
    return min(cands, key=lambda c: abs(c - fs_est))


def resample_to(x, fs_in, fs_out=FS, axis=0):
    fs_in = int(round(fs_in))
    if fs_in == fs_out:
        return x
    fr = Fraction(fs_out, fs_in).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator, axis=axis)


def highpass_1hz(x, fs, axis=-1):
    sos = butter(4, 1.0, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=axis)


def bandpass_filter(data, lowcut=1.0, highcut=50.0, fs=FS, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return lfilter(b, a, data, axis=0)


def wavelet_transform(data, wavelet="db4", level=4):
    coeffs = pywt.wavedec(data, wavelet, level=level, axis=0)
    recs = []
    for i in range(1, len(coeffs)):
        tmp = [np.zeros_like(c) for c in coeffs]; tmp[i] = coeffs[i].copy()
        r = pywt.waverec(tmp, wavelet, axis=0); recs.append(r)
    L = data.shape[0]
    recs = [r[:L] if r.shape[0] >= L else np.pad(r, ((0, L - r.shape[0]), (0, 0))) for r in recs]
    return np.stack(recs, axis=-1).astype(np.float32)


def sliding_window(arr, win=WIN, step=STEP):
    N = arr.shape[0]
    if N < win:
        return np.zeros((0, win) + arr.shape[1:], dtype=np.float32)
    return np.array([arr[i:i + win] for i in range(0, N - win + 1, step)], dtype=np.float32)


def car(a):
    """a: (T, C) 공통 평균 기준(CAR)"""
    return a - a.mean(axis=1, keepdims=True)


def trial_to_windows(a, use_car=True, window=WIN, step=STEP):
    """a: (T, C) 트라이얼 하나 (256Hz) -> (윈도우 수, 256, C, 4)"""
    a = np.asarray(a, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)           # 트라이얼 평균 제거 (DC 때문에 필터 시작이 튀는 것 방지)
    if use_car:
        a = car(a)
    a_f = bandpass_filter(a, 1.0, 50.0, FS, order=5)
    a_w = wavelet_transform(a_f)
    return sliding_window(a_w, window, step)


def band_stats(X):
    """X: (N, T, C, B) -> 대역별 평균/표준편차 (B,)"""
    B = X.shape[-1]
    flat = X.reshape(-1, B)
    return flat.mean(0).astype(np.float32), flat.std(0).astype(np.float32)


def zscore(X, mu, sd):
    B = X.shape[-1]
    return ((X.reshape(-1, B) - mu[None, :]) / (sd[None, :] + 1e-6)).reshape(X.shape).astype(np.float32)


# ---------------------------------------------------------------- PhysioNet
def find_physionet(phys_root, runs=(4, 8, 12)):
    """phys_root 아래를 뒤져서 {피험자 번호: [edf 경로...]} 를 만든다 (왼/오른 주먹 상상 run 만)."""
    pat = re.compile(r"S(\d{3})R(\d{2})\.edf$", re.I)
    out = {}
    for p in glob.glob(os.path.join(phys_root, "**", "S[0-9][0-9][0-9]R[0-9][0-9].edf"), recursive=True):
        m = pat.search(os.path.basename(p))
        if not m:
            continue
        s, r = int(m.group(1)), int(m.group(2))
        if r in runs:
            out.setdefault(s, []).append(p)
    return {s: sorted(v) for s, v in sorted(out.items())}


def load_physionet_subject(edf_paths, seg=SEG, ch_names=CH16):
    """한 피험자의 EDF들 -> (트라이얼 배열 리스트 [(T,16)], 라벨 0=왼 1=오). T1=왼 주먹, T2=오른 주먹."""
    import mne
    trials, labels = [], []
    for p in edf_paths:
        raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
        clean = [n.strip(".").strip().lower() for n in raw.ch_names]
        idx = []
        for c in ch_names:
            if c.lower() not in clean:
                raise KeyError(f"채널 {c} 없음: {os.path.basename(p)}")
            idx.append(clean.index(c.lower()))
        data = raw.get_data(picks=idx) * 1e6                 # (C, N) uV
        fs = raw.info["sfreq"]
        events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose="ERROR")
        data = resample_to(data, fs, FS, axis=1)
        data = highpass_1hz(data, FS, axis=1)
        for smp, _, ev in events:
            on = int(round(smp * FS / fs))
            a, b = on + int(seg[0] * FS), on + int(seg[1] * FS)
            if b > data.shape[1] or a < 0:
                continue
            trials.append(data[:, a:b].T.astype(np.float32))
            labels.append(0 if ev == 1 else 1)
    return trials, np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------- 내 데이터
def load_own_raw(path, n_ch=16, seg=SEG, mains=60.0, left_mk=1, right_mk=2):
    """mini_session.py 가 저장한 raw(탭 구분 35열) -> (트라이얼 리스트 [(T,n_ch)], 라벨). 앞 n_ch 채널(CH1~CH16)만 쓴다."""
    d = np.loadtxt(path, delimiter="\t")
    X = d[:, 1:1 + n_ch].T.astype(np.float64)
    ts = d[:, 33]; mk = d[:, 34]
    fs_est = 1.0 / np.median(np.diff(ts)); fs = nearest_fs(fs_est)
    nb, na = iirnotch(mains, 30, fs)
    X = filtfilt(nb, na, X, axis=1)
    X = highpass_1hz(X, fs, axis=1)
    X = resample_to(X, fs, FS, axis=1)
    trials, labels = [], []
    for i in np.where(mk != 0)[0]:
        if mk[i] not in (left_mk, right_mk):
            continue
        on = int(round(i * FS / fs))
        a, b = on + int(seg[0] * FS), on + int(seg[1] * FS)
        if b > X.shape[1]:
            continue
        trials.append(X[:, a:b].T.astype(np.float32))
        labels.append(0 if mk[i] == left_mk else 1)
    return trials, np.array(labels, dtype=np.int32)


def load_own_csv(root, subj, cls_dir="운동상상", src_fs=256, crop=None, n_ch=16):
    """<root>/<subj>/<cls_dir>/왼|오/*.csv (열 eeg_1..eeg_N) -> (트라이얼 리스트, 라벨). 앞 n_ch 열만 쓴다."""
    import pandas as pd
    base = os.path.join(root, subj, cls_dir)
    trials, labels = [], []
    for lab, folder in ((0, "왼"), (1, "오")):
        d = os.path.join(base, folder)
        if not os.path.isdir(d):
            raise SystemExit(f"폴더가 없어: {d}")
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(d, fn))
            cols = sorted([c for c in df.columns if c.startswith("eeg_")], key=lambda c: int(c.split("_")[1]))[:n_ch]
            a = df[cols].dropna().values.astype(np.float64)
            a = resample_to(a, src_fs, FS, axis=0)
            if crop:
                a = a[int(crop[0] * FS): int(crop[1] * FS)]
            trials.append(a.astype(np.float32)); labels.append(lab)
    return trials, np.array(labels, dtype=np.int32)