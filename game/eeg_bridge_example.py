"""
깃발 신호수 <-> EEG 분류기 브리지 예시

게임(index.html)은 ws://localhost:8765 에 접속해서
  - 받는 것:  {"left": 0.0~1.0, "right": 0.0~1.0}   (분류기 확률, 250ms 정도마다)
  - 보내는 것: {"type": "marker", "event": "cue", "trial": 3, "target": "left", "code": 1, "t_ms": ..., "wall": ...}

지금은 키보드로 흉내 낸 확률 대신 '무작위 확률'을 보냅니다.
classify() 부분을 실제 EEG 파이프라인(LSL 수신 -> 필터 -> CSP+LDA 등)으로 바꾸면 됩니다.

설치:  pip install websockets          (마커를 LSL로 내보내려면  pip install pylsl)
실행:  python eeg_bridge_example.py
"""
import asyncio, json, random

try:
    from pylsl import StreamInfo, StreamOutlet
    marker_outlet = StreamOutlet(StreamInfo("FlagSignalmanMarkers", "Markers", 1, 0, "string", "flag-signalman"))
except Exception:
    marker_outlet = None  # pylsl 없으면 콘솔에만 출력

import websockets


def classify():
    """여기를 실제 분류기로 교체하세요. 최근 1~2초 EEG 창으로 왼손/오른손 확률을 계산해 돌려줍니다."""
    p_left = random.uniform(0.3, 0.7)
    return p_left, 1.0 - p_left


async def handler(ws):
    print("게임 연결됨")

    async def send_probs():
        while True:
            l, r = classify()
            await ws.send(json.dumps({"left": l, "right": r}))
            await asyncio.sleep(0.25)

    sender = asyncio.create_task(send_probs())
    try:
        async for msg in ws:
            m = json.loads(msg)
            if m.get("type") == "marker":
                print(f"[marker] t={m['t_ms']:>7}ms trial={m.get('trial')} {m['event']} {m.get('target', '')}")
                if marker_outlet:
                    marker_outlet.push_sample([json.dumps(m, ensure_ascii=False)])
    finally:
        sender.cancel()
        print("게임 연결 끊김")


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("ws://localhost:8765 대기 중 — 게임에서 입력을 'EEG'로 고르고 연결을 누르세요")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
