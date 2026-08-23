#!/usr/bin/env python3
"""합성 회기 음원으로 검출·화자판별 성능을 측정하는 회귀 검증.

실제 임상 음성은 저장소에 넣지 않는다(금지선). 대신 whisper.cpp 예제 음원을
피치 변형해 두 화자를 만들고, 짧은 대상자 발화를 심은 5분 타임라인을 합성한다.

    python tests/regression.py            # 기본 임계로 검증
    python tests/regression.py --keep     # 산출물을 남긴다(디버깅용)

기준을 밑돌면 종료 코드 1을 낸다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
JFK = Path("/opt/homebrew/share/whisper-cpp/jfk.wav")

# 합격 기준 — 놓침이 치명적이므로 검출률을 가장 높게 잡는다
MIN_DETECTION = 1.00
MIN_CLASSIFICATION = 0.90
MAX_FALSE_POSITIVE_RATIO = 0.30


def resample(a: np.ndarray, ratio: float) -> np.ndarray:
    n = int(len(a) / ratio)
    return np.interp(np.arange(n) * ratio, np.arange(len(a)), a).astype(np.float32)


def pick(sig: np.ndarray, length: float, rng, min_db: float) -> np.ndarray | None:
    """실제로 소리가 들어있는 조각만 고른다 — 무음 조각은 정답이 될 수 없다."""
    for _ in range(200):
        off = rng.uniform(0, len(sig) / SR - length)
        seg = sig[int(off * SR):int((off + length) * SR)]
        if 20 * np.log10(max(np.sqrt((seg ** 2).mean()), 1e-9)) > min_db:
            return seg
    return None


def build_fixture(dst: Path, duration: float = 300.0, seed: int = 11) -> list[tuple[float, float]]:
    with wave.open(str(JFK)) as w:
        src = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    clinician, subject = src, resample(src, 0.78)      # 대상자는 낮은 음역

    x = np.random.default_rng(0).normal(0, 0.002, int(SR * duration)).astype(np.float32)

    def place(t0: float, seg: np.ndarray, amp: float) -> None:
        seg = seg * amp
        env = np.ones(len(seg))
        k = int(0.02 * SR)
        env[:k], env[-k:] = np.linspace(0, 1, k), np.linspace(1, 0, k)
        s = int(t0 * SR)
        x[s:s + len(seg)] += seg * env

    rng = np.random.default_rng(seed)
    t, truth = 4.0, []
    while t < duration - 12:
        length = rng.uniform(2.5, 5.0)
        if (seg := pick(clinician, length, rng, -35)) is not None:
            place(t, seg, 1.0)
        t += length + rng.uniform(1.5, 4.0)
        if rng.random() < 0.45 and t < duration - 12:
            length2 = rng.uniform(0.35, 1.2)            # 대상자는 짧게, 드물게
            if (seg := pick(subject, length2, rng, -28)) is not None:
                place(t, seg, 0.5)
                truth.append((round(t, 2), round(t + length2, 2)))
                t += length2 + rng.uniform(2.0, 5.0)

    wav = dst / "fixture.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=#203040:s=480x270:r=10", "-i", str(wav),
         "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(dst / "fixture.mp4")], check=True)
    return truth


def overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def score(truth: list[tuple[float, float]], segments: list[dict]) -> dict:
    detected = classified = 0
    for a, b in truth:
        hits = [s for s in segments if overlap(a, b, s["start"], s["end"]) > 0.15 * (b - a)]
        detected += bool(hits)
        classified += any(s["is_target"] for s in hits)
    targets = [s for s in segments if s["is_target"]]
    false_pos = [s for s in targets
                 if not any(overlap(s["start"], s["end"], a, b) > 0 for a, b in truth)]
    return {
        "detection": detected / len(truth),
        "classification": classified / len(truth),
        "false_positive_ratio": len(false_pos) / len(targets) if targets else 0.0,
        "n_truth": len(truth), "n_segments": len(segments), "n_targets": len(targets),
    }


def run_case(name: str, work: Path, video: Path, truth: list, extra: list[str]) -> dict:
    out = work / f"out-{name}"
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "speech_index.py"), str(video),
         "--workdir", str(out), "--no-asr", *extra],
        check=True, capture_output=True, text=True)
    segments = json.loads((out / "segments.json").read_text(encoding="utf-8"))
    return score(truth, segments)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="산출물을 남긴다")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"{tool} 가 없습니다.", file=sys.stderr)
            return 1
    if not JFK.exists():
        print(f"검증용 음원이 없습니다: {JFK}", file=sys.stderr)
        return 1

    work = Path(tempfile.mkdtemp(prefix="utterance-indexer-test-"))
    try:
        print("합성 음원 생성", flush=True)
        truth = build_fixture(work)
        video = work / "fixture.mp4"
        print(f"  대상자 발화 {len(truth)}개\n", flush=True)

        cases = {
            "자동(k-means)": [],
            "참조 1개(--target)": ["--target", f"{truth[0][0]}-{truth[0][1]}"],
        }
        failed = False
        for name, extra in cases.items():
            r = run_case(name, work, video, truth, extra)
            ok = (r["detection"] >= MIN_DETECTION
                  and r["classification"] >= MIN_CLASSIFICATION
                  and r["false_positive_ratio"] <= MAX_FALSE_POSITIVE_RATIO)
            failed |= not ok
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
            print(f"    검출 {r['detection']:.0%} · 분류 {r['classification']:.0%} · "
                  f"오탐비 {r['false_positive_ratio']:.0%} "
                  f"(구간 {r['n_segments']}개 중 대상자 {r['n_targets']}개)")
        print()
        if failed:
            print(f"기준 미달 — 검출 ≥{MIN_DETECTION:.0%}, 분류 ≥{MIN_CLASSIFICATION:.0%}, "
                  f"오탐비 ≤{MAX_FALSE_POSITIVE_RATIO:.0%}", file=sys.stderr)
        else:
            print("회귀 검증 통과")
        return 1 if failed else 0
    finally:
        if args.keep:
            print(f"\n산출물: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
