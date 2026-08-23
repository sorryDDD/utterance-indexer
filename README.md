# utterance-indexer

언어재활·진단 임상 회기를 녹화한 긴 영상에서 **누가 언제 소리를 냈는지**를 찾아, 임상가가 그
지점 사이를 건너뛰며 들을 수 있게 하는 도구다.

자발화 분석의 판단은 임상가가 귀로 한다. 이 도구는 그 판단을 대신하지 않고, **발화를 찾느라
소모되는 시간만 줄인다.** 전사는 맥락 파악용 보조 기능이다.

**전 과정이 로컬에서 수행된다.** 임상 음성·영상·전사 결과는 어떤 경로로도 외부에 나가지 않는다.

## 상태

**개발 중 — 실무 투입 불가.** 합성 음원 회귀 검증은 통과하지만(검출·분류 100%), 실제 회기
영상에서는 화자 판별이 작동하지 않았다. 현재 마일스톤 `v0.1`의 과제가 그 원인 규명이다.
자세한 것은 `AGENTS.md` §1 「현재 상태」와 `docs/manual/roadmap.md` 를 본다.

## 동작

1. ffmpeg 로 16 kHz mono 오디오를 뽑는다.
2. 에너지 VAD 와 Silero VAD 의 **합집합**으로 발성 구간을 검출한다 — 놓치는 것이 오탐보다 나쁘다.
3. 구간별 음향 임베딩(로그 스펙트럼 32밴드 + F0)으로 화자를 가른다.
4. whisper.cpp 로 전사해 각 구간에 텍스트를 붙인다(보조, 생략 가능).
5. 리뷰어·타임코드 표·자막을 낸다.

## 준비

```sh
python3 -m venv .venv && ./.venv/bin/pip install numpy
brew install ffmpeg whisper-cpp
mkdir -p ~/.cache/whisper.cpp && cd ~/.cache/whisper.cpp
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin
curl -LO https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin
```

## 실행

```sh
./.venv/bin/python src/speech_index.py <영상경로> --serve
```

브라우저에 리뷰어가 열린다. `K` / `J` 로 발화 사이를 건너뛰고, `1` `2` `3` 으로 대상자·임상가·잡음을
표시한다. 표시한 라벨을 내보내 `--labels` 로 되먹이면 화자 판별이 정확해진다.

옵션과 운용 절차는 **`docs/manual/usage.md`** 에 있다.

## 검증

```sh
./.venv/bin/python tests/regression.py     # 약 5초
markdownlint-cli2 '**/*.md' '!.venv'
```

회귀 검증은 **합성 음원만** 쓴다(금지선). 통과해도 실전 성능은 보장되지 않는다.

## 문서

| 문서 | 내용 |
|---|---|
| `AGENTS.md` | 프로젝트 고유 정보와 작업 규칙 |
| `docs/manual/working_protocol.md` | 작업 방법론의 정본 |
| `docs/manual/roadmap.md` | 마일스톤과 운용 제약 |
| `docs/manual/usage.md` | 사용법 |
| `docs/domain/language_sample_analysis_tools.md` | 자발화 분석 도구 현황 조사 |
| `docs/document_map.md` | 문서가 어디 있는가 |
