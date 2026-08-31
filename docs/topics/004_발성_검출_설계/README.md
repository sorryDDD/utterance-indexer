# 004 · 발성 검출 설계

- **상태**: 열림

## 질문

무엇을 발성으로 잡을 것인가. 놓치지 않는 것과 잡음을 들이지 않는 것 사이에서 어디에 설 것인가.

## 관련 결정

- [D-002 · 에너지 VAD와 Silero VAD의 합집합을 유지한다](../../decisions.md#d-002-vad-union)
- [D-003 · 에너지 VAD 단독 표지를 배제 게이트로 쓰지 않는다](../../decisions.md#d-003-energy-vad-gate)
- [D-004 · `f0=0`을 홀드아웃 검증 전에는 배제 게이트로 쓰지 않는다](../../decisions.md#d-004-zero-f0-gate)
- [D-005 · 참여자 비중첩 검출은 리뷰 대상 오탐 총량으로 보고한다](../../decisions.md#d-005-false-positive-reporting)
- [D-008 · F0 상태를 발성 배제 게이트로 쓰지 않는다](../../decisions.md#d-008-f0-exclusion-gate)
- [D-009 · v0.1의 상류 문제를 화자 판별로 둔다](../../decisions.md#d-009-speaker-bottleneck)

## 남은 질문

- 참여자와 겹치지 않은 검출 구간에서 잡음과 표시되지 않은 제3자 발성의 비중을 가를 수 없다.
- Silero가 놓친 대상자 발성의 성격과 국소 배경 대비를 확인하지 않았다.
- 발성 의심 등급을 둘지와 그 기준을 정하지 않았다.
- TEN VAD·pyannote VAD를 같은 실전 조건에서 시험하지 않았다.
