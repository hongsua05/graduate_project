# 운동상상 BCI 깃발 게임

뇌파(EEG)로 **왼손/오른손 운동상상**을 읽어서 게임 속 깃발을 조작하는 졸업 전시용 데모입니다.
참가자가 게임 안에서 보정(캘리브레이션)을 한 번 하면, 그 사람의 뇌파로 모델을 바로 미세조정(fine-tuning)하고 실시간으로 좌/우를 판별합니다.

<!-- TODO: 게임 화면 스크린샷 또는 GIF -->
<!-- ![demo](docs/demo.gif) -->

## 전체 구조

```
IronBCI-32 (EEG 헤드셋)
      │  시리얼(BrainFlow)
      ▼
decoder_server.py  ── WebSocket (ws://localhost:8765) ──  게임 (브라우저, index.html)
 · 실시간 전처리                                             · 보정 / 게임 진행
 · 모델 미세조정 + 추론                                       · 큐(cue) 마커 전송
```

- 게임이 보정 중에 `session_start`, `cue`(좌/우), `session_end` 마커를 WebSocket으로 보냅니다.
- 디코더는 그 시간에 맞춰 뇌파를 잘라 보정 데이터로 쓰고, 끝나면 모델을 미세조정합니다.
- 게임 모드에서는 3초 단위로 뇌파를 판별해서 결과(left / right)를 게임에 돌려줍니다.

## 모델과 처리

- **모델:** PhysioNet 운동상상 데이터로 사전학습한 Conv1D + LSTM + Attention 모델 (`ckpt_phys/`)
- **미세조정:** 게임 안 보정 데이터로 매번 사전학습 가중치에서 다시 시작해서 학습
- **전처리(보정과 실시간이 동일):**
  1. 최근 3.5초 버퍼 → 60Hz notch, 1Hz 하이패스, 256Hz로 리샘플
  2. 마지막 3초를 한 트라이얼로 사용 → 평균 제거, CAR, 1~50Hz 밴드패스
  3. db4 웨이블릿 4레벨 → 256샘플 창(step 128)으로 5개 창 생성
  4. 대역별 z-score (보정 데이터의 통계 사용)
- **판정:** 최근 창들의 확률 평균 → 스무딩 → gain/center 보정 → 0.5 기준으로 왼/오른쪽 결정 (`--thr 0`이면 "판단 안 함" 없음)

## 보정과 게임 진행

| 모드 | 트라이얼 | 설명 |
|---|---|---|
| 보정 (calib) | 26회 (왼 13 / 오른 13), 약 5분 | 화살표 큐에 맞춰 4초간 운동상상 |
| 게임 (game) | 5회 | 보정한 모델로 실시간 판별 |

보정 한 번이면 이후 게임을 계속할 수 있고, 사람이 바뀌면 보정을 다시 해야 합니다.
게임의 상상 구간은 최소 4초여야 합니다(기본값 4초).

## 필요한 것

- 하드웨어: IronBCI-32, Windows PC (시리얼 포트 예: `COM3`)
- Python 3.10 전후, 가상환경 권장
- 주요 패키지: `tensorflow`, `brainflow`, `numpy`, `scipy`, `PyWavelets`, `websockets`
  <!-- TODO: 실제 환경에서 `pip freeze > requirements.txt`로 정확한 버전 고정 -->

## 파일 구성

| 파일 | 역할 |
|---|---|
| `decoder_server.py` | 실시간 디코더 (BrainFlow 수집, 보정, 미세조정, WebSocket 서버) |
| `bci_data.py` | 데이터 전처리 함수 |
| `bci_model.py` | 모델 정의 |
| `ckpt_phys/` | PhysioNet 사전학습 가중치 (`phys_best.weights.h5`, `meta.json`) |
| `index.html` | 게임 (브라우저에서 열기) |
| `visualize.py` | 16채널 실시간 파형 확인용 (디코더와 동시에 실행 불가) |

## 설치

```bash
python -m venv mi-bci-env
mi-bci-env\Scripts\activate
pip install tensorflow brainflow numpy scipy PyWavelets websockets
```

> 가상환경 폴더를 다른 위치로 옮기면 안에 저장된 경로가 깨집니다.
> 옮겼다면 `Scripts\activate.bat`의 `VIRTUAL_ENV`를 새 경로로 고치거나 가상환경을 다시 만드세요.

## 실행 방법

1. 헤드셋을 켜고 PC에 연결합니다(포트 확인: 장치 관리자).
2. 디코더를 먼저 실행합니다.

```bash
python decoder_server.py --mode live --source brainflow --port COM3 --ckpt_dir ckpt_phys ^
  --ingame_calib --thr 0 --gain 3 --smooth 1 --step 0.25 --win_n 2 --center 0.6
```

3. 브라우저에서 `index.html`을 엽니다.
4. 게임에서 보정 모드로 시작하면 26회 보정이 끝난 뒤 모델이 자동으로 학습됩니다.
5. 학습이 끝나면 게임 모드로 진행합니다.

### 주요 옵션

| 옵션 | 설명 |
|---|---|
| `--ingame_calib` | 게임 안에서 보정 데이터를 받음 |
| `--reuse_personal` | 저장된 개인 모델을 재사용 (보정 생략) |
| `--accumulate` | 보정을 이어서 누적 (기본은 새 보정 시작 시 초기화) |
| `--personal_dir 폴더` | 개인 모델 저장 폴더 지정 |
| `--thr` | 중립 구간. `0`이면 항상 왼/오른쪽 중 하나로 판정 |
| `--gain`, `--center` | 판정 민감도와 기준점 |
| `--smooth`, `--win_n` | 스무딩 정도와 평균 낼 창 개수 |

### 실행 중 값 바꾸기

디코더 터미널에 아래처럼 입력하면 끄지 않고 값을 바꿀 수 있습니다. `?`를 입력하면 현재 값을 보여줍니다.

```
c 0.55      # center (숫자 / auto / off)
g 3         # gain
w 2         # win_n
m 1         # smooth
t 0         # thr
```

한쪽으로 치우쳐 판정될 때는 `c` 값을 조정하면 됩니다.

## 문제 해결

- **`UNABLE_TO_OPEN_PORT`**: 다른 프로그램(`visualize.py`, 이전 디코더 등)이 같은 포트를 쓰고 있습니다. 모두 종료하고 다시 실행하세요.
- **`python`이 가상환경이 아닌 Anaconda로 실행됨**: `where python`으로 확인하고, 가상환경을 먼저 활성화하세요.
- **게임은 열리는데 반응이 없음**: 디코더가 먼저 켜져 있어야 합니다(ws://localhost:8765).

## 한계

- 실시간 정확도는 아직 체계적으로 측정하지 않았습니다. 보정 데이터가 26트라이얼로 적어 사람에 따라 편차가 큽니다.
- 처음 보는 참가자(비숙련자)의 디코딩 성능은 추가 검증이 필요합니다.
- 모니터 등 주변 전자기기 잡음이 신호에 영향을 줄 수 있습니다.

## 참고

- PhysioNet EEG Motor Movement/Imagery Dataset
- 선배 연구의 모델 구조와 전처리(256Hz, 1초 창)를 이어받았습니다.
- <!-- TODO: 팀원, 지도교수, 소속 -->

## 라이선스

<!-- TODO: 라이선스 선택 (예: MIT) -->
