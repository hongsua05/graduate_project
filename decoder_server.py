# -*- coding: utf-8 -*-
"""
decoder_server.py -- 게임(index.html)에 뇌파 판정을 WebSocket 으로 보내는 서버.

게임이 받는 메시지:  {"left": 0.2, "right": 0.8}   (0~1 확률, 1초 넘게 안 오면 게임이 중립 처리)
우리가 추가로 보내는 값(게임은 무시): decision("left"/"right"/"none"), confidence, status

모드
  mock   : 가짜 값(왼/오른/중립을 번갈아). 게임 연결 테스트용. TensorFlow 필요 없음.
  live   : 진짜 판정. 보정 파일(--calib_raw) 로 PhysioNet 체크포인트를 미세조정한 뒤,
           IronBCI(BrainFlow) 또는 raw 파일 재생(replay)에서 들어오는 뇌파를 슬라이딩 윈도우로 판정.

실행 예
  python decoder_server.py --mode live --source brainflow --port COM3 --ckpt_dir ckpt_phys --ingame_calib --thr 0 --gain 3   # 게임 안에서 보정
  python decoder_server.py --mode mock                                           # 게임 연결 확인
  python decoder_server.py --mode live --source brainflow --port COM3 \\
        --ckpt_dir ckpt_phys --calib_raw 보정세션_raw.txt                          # 진짜
  python decoder_server.py --mode live --source replay --raw 다른세션_raw.txt \\
        --ckpt_dir ckpt_phys --calib_raw 보정세션_raw.txt --speed 4                # 오프라인 리허설

판정 방식(학습 때와 같은 처리)
  최근 3.5초를 버퍼에 두고 -> 60Hz 노치 -> 1Hz 하이패스 -> 256Hz 로 리샘플 -> 마지막 3초를 트라이얼처럼 처리
  -> 1초 윈도우 5개 확률 평균 -> 최근 N번 평균(누적) -> 확신이 낮으면 0.5/0.5(중립)
"""
import os, sys, json, time, argparse, asyncio, threading, collections, csv
import numpy as np

STATE = {"left": 0.5, "right": 0.5, "decision": "none", "confidence": 0.0, "status": "starting", "t": 0.0}
LOCK = threading.Lock()


def set_state(**kw):
    with LOCK:
        STATE.update(kw); STATE["t"] = time.time()


def get_state():
    with LOCK:
        return dict(STATE)


# ------------------------------------------------------------------ mock
def mock_loop(a, stop):
    seq = [("left", 4.0), ("none", 2.0), ("right", 4.0), ("none", 2.0)]
    rng = np.random.RandomState(0); i = 0
    while not stop.is_set():
        d, dur = seq[i % len(seq)]; t0 = time.time(); i += 1
        while time.time() - t0 < dur and not stop.is_set():
            if d == "none": r = 0.5
            else:
                base = 0.5 + a.mock_amp if d == "right" else 0.5 - a.mock_amp
                r = float(np.clip(base + rng.randn() * a.mock_noise, 0.02, 0.98))
            set_state(left=1 - r, right=r, decision=d, confidence=abs(r - 0.5) * 2, status="mock")
            time.sleep(0.2)


# ------------------------------------------------------------------ 입력원
class ReplaySource:
    """raw 파일(mini_session 형식)을 실시간처럼 흘려준다 (리허설/테스트용)."""
    def __init__(self, path, speed=1.0, n_ch=16):
        d = np.loadtxt(path, delimiter="\t")
        self.X = d[:, 1:1 + n_ch].T.astype(np.float64); ts = d[:, 33]
        self.fs = int(round(1.0 / np.median(np.diff(ts)))); self.speed = speed; self.pos = 0; self.t0 = time.time()
    def read(self):
        due = int((time.time() - self.t0) * self.fs * self.speed)
        if due <= self.pos:
            time.sleep(0.02); return np.zeros((self.X.shape[0], 0))
        out = self.X[:, self.pos:due]; self.pos = min(due, self.X.shape[1])
        if self.pos >= self.X.shape[1] and out.shape[1] == 0: raise EOFError
        return out
    def close(self): pass


class BrainFlowSource:
    def __init__(self, port, synthetic=False, n_ch=16):
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        try:
            import config
        except ImportError:
            config = None
        params = BrainFlowInputParams()
        if synthetic: bid = BoardIds.SYNTHETIC_BOARD
        else:
            bid = getattr(config, "BOARD_ID", BoardIds.IRONBCI_32_BOARD); params.serial_port = port or getattr(config, "SERIAL_PORT", "COM3")
        BoardShim.disable_board_logger()
        self.board = BoardShim(bid, params); self.board.prepare_session(); self.board.start_stream(450000)
        self.chs = list(BoardShim.get_eeg_channels(bid))[:n_ch]; nominal = BoardShim.get_sampling_rate(bid)
        ts_ch = BoardShim.get_timestamp_channel(bid); time.sleep(2.5)       # 실제 샘플링 속도를 시각 열로 직접 잰다
        d = self.board.get_current_board_data(1500); fs_m = None
        if d.shape[1] > 100:
            dt = np.diff(d[ts_ch]); dt = dt[dt > 0]
            if len(dt): fs_m = 1.0 / float(np.median(dt))
        cands = [100, 128, 160, 200, 250, 256, 500, 512, 1000, 1024]
        self.fs = min(cands, key=lambda c: abs(c - fs_m)) if fs_m else nominal
        print(f"[입력] 장비가 말하는 샘플링 {nominal}Hz, 실제로 잰 값 {fs_m if fs_m else float('nan'):.1f}Hz -> {self.fs}Hz 로 처리")
    def read(self):
        data = self.board.get_board_data()
        if data.shape[1] == 0: time.sleep(0.02); return np.zeros((len(self.chs), 0))
        return data[self.chs, :].astype(np.float64)
    def close(self):
        try: self.board.stop_stream(); self.board.release_session()
        except Exception: pass


# ------------------------------------------------------------------ 보정 + 판정
def finetune(m, a, T, y, use_car):
    """트라이얼 리스트 T(각 (768,16), 256Hz)와 라벨 y(0=왼,1=오)로 미세조정. (mu, sd) 반환."""
    import tensorflow as tf
    from bci_data import trial_to_windows, band_stats, zscore
    from bci_model import augment_numpy
    y = np.array(y, np.int32)
    W = [trial_to_windows(t, use_car=use_car) for t in T]
    X = np.concatenate(W); yy = np.concatenate([np.full(len(w), l, np.float32) for w, l in zip(W, y)])
    mu, sd = band_stats(X); X = zscore(X, mu, sd)
    X, yy = augment_numpy(X, yy.astype(np.int32), times=1, seed=42); yy = yy.astype(np.float32)
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=a.ft_lr, clipnorm=1.0),
              loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05), metrics=["accuracy"])
    h = m.fit(X, yy, epochs=a.ft_epochs, batch_size=32, verbose=0, shuffle=True)
    print(f"[보정] 트라이얼 {len(y)}개로 미세조정 완료 (학습 정확도 {h.history['accuracy'][-1]*100:.0f}%)")
    m.save_weights(os.path.join(a.personal_dir, "personal.weights.h5")); np.savez(os.path.join(a.personal_dir, "stats.npz"), mu=mu, sd=sd)
    return mu, sd


def prepare_model(a):
    """체크포인트를 불러와 --calib_raw 로 미세조정. (모델, mu, sd, use_car) 반환. 결과는 personal_dir 에 저장.
    --ingame_calib 이면 파일 없이 시작하고, 게임 안 캘리브레이션이 끝나면 그때 미세조정한다 (mu, sd 는 None)."""
    from bci_data import load_own_raw, CH16, WIN
    from bci_model import build_hybrid_model
    meta = json.load(open(os.path.join(a.ckpt_dir, "meta.json"))); use_car = meta.get("use_car", True)
    m = build_hybrid_model(WIN, len(CH16), meta["bands"], heads=meta["heads"], l2=meta["l2"])
    m.load_weights(os.path.join(a.ckpt_dir, meta["weights"]))
    os.makedirs(a.personal_dir, exist_ok=True)
    stats_f = os.path.join(a.personal_dir, "stats.npz")
    if a.calib_raw:
        T, y = [], []
        for p in a.calib_raw.split(","):
            t, l = load_own_raw(p.strip(), n_ch=len(CH16)); T += t; y += list(l)
        y = np.array(y, np.int32)
        if len(y) < 10 or min((y == 0).sum(), (y == 1).sum()) < 4:
            raise SystemExit(f"보정 트라이얼이 부족해 (왼 {(y==0).sum()}, 오 {(y==1).sum()}). 최소 20개 권장.")
        if a.calib_n: # 앞에서부터 N개만
            T, y = T[:a.calib_n], y[:a.calib_n]
        mu, sd = finetune(m, a, T, y, use_car)
    elif a.ingame_calib and getattr(a, "reuse_personal", False) and os.path.exists(stats_f):
        m.load_weights(os.path.join(a.personal_dir, "personal.weights.h5")); z = np.load(stats_f); mu, sd = z["mu"], z["sd"]
        print("[보정] 저번에 저장해 둔 개인 모델을 불러옴 (--reuse_personal). 바로 게임을 해도 돼")
    elif a.ingame_calib:
        mu = sd = None
        if not getattr(a, "quiet", False): print("[보정] 게임 안 캘리브레이션을 기다리는 중 (게임에서 '캘리브레이션' 모드로 시작하세요)")
    elif os.path.exists(stats_f):
        m.load_weights(os.path.join(a.personal_dir, "personal.weights.h5")); z = np.load(stats_f); mu, sd = z["mu"], z["sd"]
        print("[보정] 저장된 개인 모델을 불러옴")
    else:
        raise SystemExit("--calib_raw 가 필요해 (저장된 개인 모델도 없음). 게임 안에서 보정하려면 --ingame_calib")
    return m, mu, sd, use_car


# ------------------------------------------------------------------ 게임 안 캘리브레이션
class InGameCalib:
    """게임이 보내는 마커(session_start / cue / session_end)와 같은 시각에 받은 EEG 로 보정 트라이얼을 만든다."""
    def __init__(self, fs):
        self.fs = fs; self.lock = threading.Lock()
        self.chunks = []                      # (청크 끝 시각, 그때까지 받은 총 샘플 수)
        self.data = []                        # 받은 EEG 조각들 (16, n)
        self.total = 0; self.active = False; self.cues = []   # (큐 시각, 라벨 0/1)
        self.T = []; self.y = []; self.accumulate = False   # 보정 트라이얼 (--accumulate 일 때만 여러 번 해도 쌓임)
        self.todo = False; self.sessions = 0

    def feed(self, chunk):
        with self.lock:
            if not self.active: return
            self.data.append(chunk.copy()); self.total += chunk.shape[1]; self.chunks.append((time.time(), self.total))

    def on_message(self, msg):
        try: d = json.loads(msg)
        except Exception: return
        if d.get("type") != "marker": return
        ev = d.get("event"); t = time.time()
        with self.lock:
            if ev == "session_start":
                self.active = d.get("mode") == "calib"
                if self.active:
                    self.data, self.chunks, self.total, self.cues = [], [], 0, []
                    if not self.accumulate: self.T, self.y = [], []     # 새 사람(새 보정)이면 이전 기록은 버린다
                    print("[보정] 게임 캘리브레이션 시작 -- EEG 기록 중")
            elif ev == "cue" and self.active and d.get("target") in ("left", "right"):
                self.cues.append((t, 0 if d["target"] == "left" else 1))
            elif ev == "session_end" and self.active:
                self.active = False
                if d.get("aborted"): print("[보정] 중간에 끝나서 이번 기록은 버려")
                else: self.todo = True

    def build(self):
        """기록한 EEG 를 학습 때와 똑같이 처리해 트라이얼로 자른다 -> (새 트라이얼, 새 라벨)"""
        from bci_data import FS, highpass_1hz, resample_to
        from scipy.signal import iirnotch, filtfilt
        with self.lock:
            X = np.concatenate(self.data, axis=1) if self.data else np.zeros((16, 0)); chunks, cues = list(self.chunks), list(self.cues)
        if X.shape[1] < 10 * self.fs or not cues: return [], []
        nb, na = iirnotch(self.mains, 30, self.fs)
        X = filtfilt(nb, na, X, axis=1); X = highpass_1hz(X, self.fs, axis=1); X = resample_to(X, self.fs, FS, axis=1)
        tends = np.array([c[0] for c in chunks]); counts = np.array([c[1] for c in chunks])
        T, y = [], []
        for tc, lab in cues:
            k = int(np.searchsorted(tends, tc)); k = min(k, len(tends) - 1)
            idx = counts[k] - (tends[k] - tc) * self.fs          # 큐 시각의 샘플 번호
            on = int(round(idx * FS / self.fs)); a_, b_ = on + int(0.5 * FS), on + int(3.5 * FS)
            if a_ < 0 or b_ > X.shape[1]: continue
            T.append(X[:, a_:b_].T.astype(np.float32)); y.append(lab)
        return T, y


class LiveDecoder:
    def __init__(self, a, model, mu, sd, use_car, fs):
        from bci_data import FS, resample_to, highpass_1hz, trial_to_windows, zscore
        from scipy.signal import iirnotch, filtfilt
        self.a, self.model, self.mu, self.sd, self.use_car, self.fs = a, model, mu, sd, use_car, fs
        self.FS, self.resample_to, self.hp, self.ttw, self.zs = FS, resample_to, highpass_1hz, trial_to_windows, zscore
        self.notch = iirnotch(a.mains, 30, fs); self.filtfilt = filtfilt
        self.n_seen = 0; self.buf = np.zeros((16, 0)); self.need = int(round(3.5 * fs)); self.since = 0
        self.probs = collections.deque(maxlen=a.smooth); self.step = int(round(a.step * fs))
        self.hist = collections.deque(maxlen=max(10, int(a.center_sec / a.step))); self.center = 0.5

    def feed(self, chunk):
        self.n_seen += chunk.shape[1]
        self.buf = np.concatenate([self.buf, chunk], axis=1)[:, -int(5 * self.fs):]; self.since += chunk.shape[1]
        if self.buf.shape[1] < self.need:
            set_state(status="warming_up", left=.5, right=.5, decision="none", confidence=0.0); return None
        if self.since < self.step: return None
        self.since = 0
        x = self.buf[:, -self.need:]
        x = self.filtfilt(self.notch[0], self.notch[1], x, axis=1); x = self.hp(x, self.fs, axis=1)
        x = self.resample_to(x, self.fs, self.FS, axis=1)[:, -768:]
        W = self.ttw(x.T.astype(np.float32), use_car=self.use_car)
        Z = self.zs(W, self.mu, self.sd); self.zabs = float(np.abs(Z).mean())
        pw = self.model.predict(Z, batch_size=8, verbose=0).reshape(-1)
        p = float(pw[-self.a.win_n:].mean())   # 최근 win_n 개 윈도우만 평균 (작을수록 빠르게 반응)
        self.probs.append(p); self.hist.append(p); ps = float(np.mean(self.probs))
        # 한쪽으로 쏠림 보정: 'auto' = 최근 center_sec 초 판정의 중앙값을 0.5 로 본다 / 숫자 = 그 값을 고정 기준으로 / off = 보정 없음
        c = self.a.center
        if c == "auto": self.center = float(np.median(self.hist)) if len(self.hist) >= 8 else 0.5
        elif c != "off": self.center = float(c)
        else: self.center = 0.5
        ps = float(np.clip(0.5 + self.a.gain * (ps - self.center), 0.0, 1.0)); conf = abs(ps - 0.5) * 2   # gain: 기준에서 벗어난 만큼을 몇 배로 키워 게임에 보냄
        if abs(ps - 0.5) < self.a.thr: l = r = 0.5; dec = "none"
        else: r = ps; l = 1 - ps; dec = "right" if ps > 0.5 else "left"
        set_state(left=l, right=r, decision=dec, confidence=conf, status="live")
        return p, ps, dec, self.n_seen / self.fs, self.zabs, self.center


CALIB = None


def tune_thread(a, dec_box, stop):
    """디코더가 켜진 채로 터미널에서 값을 바꾼다.  예)  c 0.58 [엔터]   g 4   w 3   m 2   t 0.05
       c=center(기준선)  g=gain  w=win_n(최근 윈도우 몇 개)  m=smooth  t=thr(중립폭)  ? = 현재 값"""
    def show(): print(f"[설정] center={a.center} gain={a.gain} win_n={a.win_n} smooth={a.smooth} thr={a.thr}")
    while not stop.is_set():
        try: line = sys.stdin.readline()
        except Exception: return
        if line == "": return                               # 입력창이 없으면 조용히 끝
        parts = line.strip().split()
        if not parts: continue
        k = parts[0].lower()
        try:
            if k == "?": show(); continue
            v = parts[1]
            if k == "c": a.center = "off" if v == "off" else ("auto" if v == "auto" else float(v))
            elif k == "g": a.gain = float(v)
            elif k == "w": a.win_n = max(1, min(5, int(v)))
            elif k == "m":
                a.smooth = max(1, int(v))
                if dec_box: dec_box[0].probs = collections.deque(list(dec_box[0].probs)[-a.smooth:], maxlen=a.smooth)
            elif k == "t": a.thr = float(v)
            else: print("[설정] c / g / w / m / t 중 하나로 입력해. 예) c 0.58"); continue
            show()
        except Exception:
            print("[설정] 값이 이상해. 예) c 0.58")


def live_loop(a, stop):
    global CALIB
    model, mu, sd, use_car = prepare_model(a)
    src = ReplaySource(a.raw, a.speed) if a.source == "replay" else BrainFlowSource(a.port, a.source == "synthetic")
    dec = LiveDecoder(a, model, mu, sd, use_car, src.fs)
    threading.Thread(target=tune_thread, args=(a, [dec], stop), daemon=True).start()
    print("[설정] 켜진 채로 값 바꾸기: 'c 0.58' 엔터 (기준선) / 'g 4' (gain) / 'w 3' / 'm 2' / 't 0.05' / '?' (현재 값)")
    if a.ingame_calib: CALIB = InGameCalib(src.fs); CALIB.mains = a.mains; CALIB.accumulate = a.accumulate
    logf = open(a.log, "w", newline="") if a.log else None
    w = csv.writer(logf) if logf else None
    if w: w.writerow(["unix_time", "p_right_raw", "p_right_smoothed", "decision", "data_t"])
    set_state(status="warming_up" if dec.mu is not None else "need_calibration"); print("[준비 완료] 판정 시작" if dec.mu is not None else "[준비 완료] 캘리브레이션 대기")
    try:
        while not stop.is_set():
            try: chunk = src.read()
            except EOFError: print("[끝] 재생 파일 끝"); break
            if chunk.shape[1] == 0: continue
            if CALIB:
                CALIB.feed(chunk)
                if CALIB.todo:                                    # 게임 캘리브레이션이 끝났다 -> 미세조정
                    CALIB.todo = False; set_state(status="calibrating", left=.5, right=.5, decision="none", confidence=0.0)
                    T, y = CALIB.build(); print(f"[보정] 이번 기록에서 트라이얼 {len(y)}개 (왼 {y.count(0)}, 오 {y.count(1)})")
                    CALIB.T += T; CALIB.y += y
                    if len(CALIB.y) >= 10 and min(CALIB.y.count(0), CALIB.y.count(1)) >= 4:
                        print(f"[보정] 지금까지 모인 {len(CALIB.y)}개로 미세조정하는 중... (잠깐 기다려, 게임은 그대로 두세요)")
                        m2, _, _, _ = prepare_model(argparse.Namespace(**{**vars(a), "calib_raw": None, "ingame_calib": True, "quiet": True}))
                        dec.mu, dec.sd = finetune(m2, a, CALIB.T, CALIB.y, use_car); dec.model = m2
                        dec.buf = np.zeros((16, 0)); dec.since = 0; dec.probs.clear(); dec.hist.clear()
                        print("[보정] 끝! 게임에서 '게임' 모드로 시작하면 돼"); set_state(status="warming_up")
                    else:
                        print("[보정] 아직 부족해 (왼/오 각 4개 이상, 총 10개 이상 필요). 캘리브레이션을 한 번 더 하면 쌓여서 같이 학습해")
                        set_state(status="need_calibration")
                    continue
                if dec.mu is None or CALIB.active: continue      # 보정 전이거나 보정 중이면 판정 안 함
            out = dec.feed(chunk)
            if out:
                print(f"p(right)={out[0]:.2f}  누적={out[1]:.2f}  -> {out[2]}   (입력크기 |z|={out[4]:.1f}, 정상은 1 안팎 / 기준 {out[5]:.2f})")
                if w: w.writerow([f"{time.time():.3f}", f"{out[0]:.4f}", f"{out[1]:.4f}", out[2], f"{out[3]:.3f}"]); logf.flush()
    finally:
        src.close()
        if logf: logf.close()
    set_state(status="stopped", left=.5, right=.5, decision="none", confidence=0.0)
    if a.source == "replay":          # 재생 리허설은 파일이 끝나면 서버도 같이 종료
        time.sleep(1.0); stop.set()


# ------------------------------------------------------------------ WebSocket
async def serve(a, stop):
    import websockets
    evf = open(a.events_log, "a", encoding="utf-8", buffering=1) if a.events_log else None
    async def handler(ws):
        async def reader():                       # 게임이 보내는 이벤트 마커를 읽어서 기록 (안 읽으면 연결이 끊길 수 있음)
            try:
                async for msg in ws:
                    if evf: evf.write(f"{time.time():.3f}\t{get_state().get('right', 0.5):.3f}\t{msg}\n")
                    if CALIB: CALIB.on_message(msg)
            except Exception:
                pass
        rt = asyncio.ensure_future(reader())
        print("[연결] 게임이 접속했어")
        try:
            while True:
                await ws.send(json.dumps(get_state())); await asyncio.sleep(a.send_every)
        except Exception:
            pass
        finally:
            rt.cancel(); print("[연결] 게임 접속이 끊겼어")
    async with websockets.serve(handler, a.host, a.ws_port):
        print(f"[서버] ws://localhost:{a.ws_port}  (게임의 WebSocket 주소 칸에 이걸 넣으세요)")
        while not stop.is_set(): await asyncio.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "live"], default="mock")
    ap.add_argument("--source", choices=["brainflow", "replay", "synthetic"], default="brainflow")
    ap.add_argument("--port", default=None, help="IronBCI 시리얼 포트 (예: COM3)")
    ap.add_argument("--raw", help="replay 에 쓸 raw 파일")
    ap.add_argument("--speed", type=float, default=1.0, help="replay 배속")
    ap.add_argument("--ckpt_dir", default="ckpt_phys"); ap.add_argument("--calib_raw", help="보정 세션 raw (쉼표로 여러 개)")
    ap.add_argument("--calib_n", type=int, default=0, help="보정에 쓸 트라이얼 수(앞에서부터). 0=전부")
    ap.add_argument("--personal_dir", default="personal"); ap.add_argument("--ft_epochs", type=int, default=20); ap.add_argument("--ft_lr", type=float, default=5e-5)
    ap.add_argument("--ws_port", type=int, default=8765); ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--send_every", type=float, default=0.1)
    ap.add_argument("--step", type=float, default=0.5, help="판정 간격(초)"); ap.add_argument("--smooth", type=int, default=4, help="최근 몇 번의 판정을 평균낼지")
    ap.add_argument("--thr", type=float, default=0.08, help="|누적확률-0.5| 가 이보다 작으면 중립")
    ap.add_argument("--mock_amp", type=float, default=0.3, help="mock 모드에서 0.5 로부터 벌어지는 크기 (0.3=강함, 0.1=실제 모델처럼 약함)")
    ap.add_argument("--mock_noise", type=float, default=0.06)
    ap.add_argument("--events_log", default="game_events.tsv", help="게임이 보낸 이벤트 마커 기록 (시각, 그때의 p, 내용)")
    ap.add_argument("--center", default="off", help="쏠림 보정: auto(최근 판정 중앙값을 기준으로) / 숫자(예: 0.65, 그 값을 기준으로) / off")
    ap.add_argument("--center_sec", type=float, default=30.0, help="auto 일 때 기준을 구하는 최근 시간(초)")
    ap.add_argument("--win_n", type=int, default=5, help="트라이얼 윈도우 5개 중 최근 몇 개만 평균낼지. 5=전부(안정), 1~2=빠른 반응(흔들림 큼)")
    ap.add_argument("--gain", type=float, default=1.0, help="기준(center)에서 벗어난 정도를 몇 배로 키울지. 1=그대로, 3=3배 (작은 치우침도 게임이 크게 느낌)")
    ap.add_argument("--ingame_calib", action="store_true", help="보정 데이터를 mini_session 대신 게임 안 캘리브레이션 모드에서 받는다 (--calib_raw 불필요)")
    ap.add_argument("--reuse_personal", action="store_true", help="--ingame_calib 일 때, 저번에 저장된 개인 모델(personal 폴더)이 있으면 그걸로 시작")
    ap.add_argument("--accumulate", action="store_true", help="--ingame_calib 에서 캘리브레이션을 여러 번 하면 기록을 쌓아서 같이 학습 (같은 사람일 때만!). 기본은 매번 새로 시작")
    ap.add_argument("--mains", type=float, default=60.0); ap.add_argument("--log", default="decoder_log.csv")
    a = ap.parse_args()
    stop = threading.Event()
    target = (lambda: mock_loop(a, stop)) if a.mode == "mock" else (lambda: live_loop(a, stop))
    th = threading.Thread(target=target, daemon=True); th.start()
    try: asyncio.run(serve(a, stop))
    except KeyboardInterrupt: pass
    finally: stop.set()


if __name__ == "__main__":
    main()