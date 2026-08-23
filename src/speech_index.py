#!/usr/bin/env python3
"""회기 영상에서 '누가 · 언제' 소리를 냈는지 인덱싱한다.

임상가가 귀로 듣고 판단하므로, 이 도구의 목적은 전사가 아니라 **탐색 시간 단축**이다.
발성 구간을 빠짐없이 찾아 화자를 갈라 두고, 그 지점 사이를 점프하며 들을 수 있는
리뷰어를 만든다. 전사는 맥락 파악용 보조 정보로만 붙는다.

전 과정이 로컬에서 수행된다(ffmpeg + whisper.cpp + numpy). 외부 전송은 없다.

    1) ffmpeg 로 16 kHz mono 추출
    2) 발성 구간 검출 — 에너지 VAD 와 Silero VAD 의 **합집합**
       (놓치는 것이 오탐보다 나쁘므로 합집합을 기본으로 한다)
    3) 구간별 음향 임베딩(로그 스펙트럼 32밴드 + F0)으로 화자 분리
    4) whisper.cpp 전사를 구간에 부착(선택)
    5) reviewer.html · index.md · segments.json · subtitles.vtt 생성
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 16000
FRAME = 400          # 25 ms
HOP = 160            # 10 ms
N_BANDS = 32
TEMPLATE = Path(__file__).with_name("reviewer_template.html")


# --------------------------------------------------------------------------- utils

def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def hhmmss(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def vtt_ts(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def parse_ts(text: str) -> float:
    parts = text.strip().split(":")
    if not all(p.replace(".", "", 1).replace(",", "", 1).isdigit() for p in parts):
        raise ValueError(f"시각 형식이 올바르지 않습니다: {text!r}")
    sec = 0.0
    for p in parts:
        sec = sec * 60 + float(p.replace(",", "."))
    return sec


def parse_range(text: str) -> tuple[float, float]:
    if "-" not in text:
        raise ValueError(f"구간은 '시작-끝' 형식이어야 합니다: {text!r}")
    a, b = text.split("-", 1)
    return parse_ts(a), parse_ts(b)


# --------------------------------------------------------------------------- audio

def extract_audio(video: Path, wav: Path) -> None:
    if wav.exists():
        return
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav)])


def load_wav(wav: Path) -> np.ndarray:
    with wave.open(str(wav), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def probe_height(video: Path) -> int | None:
    try:
        out = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-show_entries", "stream=height", "-of", "csv=p=0", str(video)]).stdout
        return int(out.strip().splitlines()[0])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return None


def make_proxy(video: Path, dst: Path, height: int = 480) -> None:
    """브라우저에서 확실히 재생·탐색되는 저용량 H.264 사본.

    원본이 목표 높이보다 작으면 그대로 둔다. 키워봐야 화질은 그대로이고 용량만 는다.
    """
    if dst.exists():
        return
    src_h = probe_height(video)
    target = min(height, src_h) if src_h else height
    target -= target % 2                      # H.264 는 짝수 해상도를 요구한다
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", f"scale=-2:{target}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "28", "-g", "48", "-c:a", "aac", "-b:a", "96k",
         "-movflags", "+faststart", str(dst)])


# --------------------------------------------------------------------------- VAD

@dataclass
class Segment:
    idx: int
    start: float
    end: float
    src: str                     # E=에너지, S=Silero, ES=둘 다
    rms_db: float = 0.0
    f0: float = 0.0
    embedding: list[float] = field(default_factory=list, repr=False)
    speaker: int | None = None
    score: float = 0.0
    is_target: bool = False
    text: str = ""

    @property
    def dur(self) -> float:
        return self.end - self.start


def frame_signal(x: np.ndarray) -> np.ndarray:
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx] * np.hanning(FRAME)[None, :]


def energy_vad(x: np.ndarray, *, min_dur: float, max_gap: float,
               sensitivity: float) -> list[tuple[float, float]]:
    frames = frame_signal(x)
    db = 10 * np.log10(np.maximum((frames ** 2).mean(axis=1), 1e-12))
    floor, peak = np.percentile(db, 20), np.percentile(db, 95)
    voiced = db > floor + (peak - floor) * sensitivity

    spans, start, silence = [], None, 0
    gap_frames = int(max_gap / (HOP / SR))
    for i, val in enumerate(voiced):
        if val:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence > gap_frames:
                spans.append((start * HOP / SR, (i - silence) * HOP / SR))
                start = None
    if start is not None:
        spans.append((start * HOP / SR, len(voiced) * HOP / SR))
    return [(a, b) for a, b in spans if b - a >= min_dur]


def silero_vad(wav: Path, model: Path, *, threshold: float, min_dur: float,
               min_silence: float, pad: float) -> list[tuple[float, float]]:
    """whisper.cpp 내장 Silero VAD. GPU 경로는 이 빌드에서 불안정하여 CPU 로 돌린다."""
    binary = shutil.which("whisper-vad-speech-segments")
    if not binary or not model.exists():
        return []
    try:
        proc = subprocess.run(
            [binary, "-f", str(wav), "-vm", str(model), "-vt", str(threshold),
             "-vspd", str(int(min_dur * 1000)), "-vsd", str(int(min_silence * 1000)),
             "-vp", str(int(pad * 1000))],
            capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return []
    out = proc.stdout + proc.stderr
    spans = []
    for m in re.finditer(r"Speech segment \d+: start = ([\d.]+), end = ([\d.]+)", out):
        spans.append((float(m.group(1)) / 100.0, float(m.group(2)) / 100.0))
    return spans


def merge_spans(energy: list[tuple[float, float]], silero: list[tuple[float, float]],
                *, join_gap: float, min_dur: float) -> list[tuple[float, float, str]]:
    """두 검출 결과의 합집합.

    통합은 구간이 실제로 겹칠 때(또는 join_gap 이내로 맞닿을 때)만 한다. 간격이 넓은
    이웃까지 묶으면 화자가 바뀌는 지점이 한 구간으로 뭉쳐 '누가'를 잃는다.
    """
    tagged = [(a, b, "E") for a, b in energy] + [(a, b, "S") for a, b in silero]
    if not tagged:
        return []
    tagged.sort()
    merged: list[list] = [list(tagged[0])]
    for a, b, src in tagged[1:]:
        last = merged[-1]
        if a - last[1] <= join_gap:
            last[1] = max(last[1], b)
            if src not in last[2]:
                last[2] = "ES"
        else:
            merged.append([a, b, src])
    return [(a, b, s) for a, b, s in merged if b - a >= min_dur]


# --------------------------------------------------------------------------- embedding

def _band_edges() -> np.ndarray:
    mel = lambda f: 2595 * math.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    freqs = imel(np.linspace(mel(80.0), mel(7600.0), N_BANDS + 1))
    return (freqs / (SR / 2) * (FRAME // 2)).astype(int)


BAND_EDGES = _band_edges()


def estimate_f0(seg: np.ndarray) -> float:
    f0s, win = [], 1024
    for i in range(0, max(1, len(seg) - win), win // 2):
        chunk = seg[i:i + win]
        if len(chunk) < win or float(np.sqrt((chunk ** 2).mean())) < 1e-3:
            continue
        c = chunk - chunk.mean()
        ac = np.correlate(c, c, mode="full")[win - 1:]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        lo, hi = SR // 400, SR // 60          # 60–400 Hz
        if hi >= len(ac):
            continue
        peak = int(np.argmax(ac[lo:hi])) + lo
        if ac[peak] > 0.3:
            f0s.append(SR / peak)
    return float(np.median(f0s)) if f0s else 0.0


def embed(seg_audio: np.ndarray) -> tuple[np.ndarray, float, float]:
    frames = frame_signal(seg_audio)
    if len(frames) == 0:
        return np.zeros(N_BANDS * 2 + 1), -99.0, 0.0
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    bands = np.stack([
        spec[:, BAND_EDGES[i]:max(BAND_EDGES[i] + 1, BAND_EDGES[i + 1])].mean(axis=1)
        for i in range(N_BANDS)], axis=1)
    log_bands = np.log(bands + 1e-10)
    log_bands -= log_bands.mean(axis=1, keepdims=True)      # 볼륨·채널 정규화
    f0 = estimate_f0(seg_audio)
    vec = np.concatenate([log_bands.mean(axis=0), log_bands.std(axis=0), [f0 / 200.0]])
    rms_db = float(10 * np.log10(max((seg_audio ** 2).mean(), 1e-12)))
    return vec, rms_db, f0


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# --------------------------------------------------------------------------- speaker

def kmeans(X: np.ndarray, k: int = 2, iters: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if (new == labels).all():
            break
        labels = new
        for j in range(k):
            if (labels == j).any():
                c[j] = X[labels == j].mean(axis=0)
    return labels


def split_by_similarity(scores: np.ndarray) -> float:
    """유사도 분포를 두 무리로 갈라 임계값을 정한다(1차원 k-means)."""
    lab = kmeans(scores.reshape(-1, 1), 2)
    hi = int(np.argmax([scores[lab == j].mean() if (lab == j).any() else -9 for j in (0, 1)]))
    high, low = scores[lab == hi], scores[lab != hi]
    if not len(low):
        return float(scores.min())
    return float((high.min() + low.max()) / 2)


# --------------------------------------------------------------------------- ASR

def find_whisper() -> str | None:
    for name in ("whisper-cli", "whisper-cpp", "whisper"):
        if (p := shutil.which(name)):
            return p
    return None


def default_model(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser()
    for base in (Path.home() / ".cache/whisper.cpp", Path("/opt/homebrew/share/whisper-cpp")):
        if base.is_dir():
            hits = sorted(p for p in base.glob("ggml-*.bin") if "silero" not in p.name)
            if hits:
                return hits[-1]
    return None


def transcribe(wav: Path, out_json: Path, model: Path, binary: str,
               lang: str, prompt: str | None) -> None:
    if out_json.exists():
        return
    cmd = [binary, "-m", str(model), "-f", str(wav), "-l", lang,
           "-oj", "-of", str(out_json.with_suffix("")),
           "-t", str(os.cpu_count() or 4), "-pp"]
    if prompt:
        cmd += ["--prompt", prompt]
    run(cmd)


def load_asr(out_json: Path) -> list[tuple[float, float, str]]:
    if not out_json.exists():
        return []
    data = json.loads(out_json.read_text(encoding="utf-8"))
    items = []
    for seg in data.get("transcription", []):
        off = seg.get("offsets", {})
        items.append((off.get("from", 0) / 1000.0, off.get("to", 0) / 1000.0,
                      seg.get("text", "").strip()))
    return items


def attach_text(segments: list[Segment], asr: list[tuple[float, float, str]]) -> None:
    for s in segments:
        parts = [t for a, b, t in asr if b > s.start and a < s.end and t]
        s.text = " ".join(parts)[:120]


# --------------------------------------------------------------------------- output

def write_reviewer(dst: Path, video_src: str, name: str, duration: float,
                   segments: list[Segment], pad: float) -> None:
    data = {
        "key": name, "name": name, "video": video_src, "duration": duration, "pad": pad,
        "segments": [{"idx": s.idx, "start": round(s.start, 2), "end": round(s.end, 2),
                      "dur": round(s.dur, 2), "is_target": s.is_target,
                      "score": round(s.score, 3), "f0": round(s.f0),
                      "src": s.src, "text": s.text} for s in segments],
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    dst.write_text(html, encoding="utf-8")


def write_vtt(dst: Path, segments: list[Segment]) -> None:
    lines = ["WEBVTT", ""]
    for s in segments:
        who = "대상자" if s.is_target else "임상가"
        lines += [f"{vtt_ts(s.start)} --> {vtt_ts(s.end)}",
                  f"[{who}] {s.text}".rstrip(), ""]
    dst.write_text("\n".join(lines), encoding="utf-8")


def write_index(dst: Path, name: str, duration: float, segments: list[Segment],
                mode: str) -> None:
    targets = [s for s in segments if s.is_target]
    voiced = sum(s.dur for s in segments)
    lines = [
        f"# 발화 인덱스 — {name}", "",
        f"- 원본 길이: **{hhmmss(duration)}**",
        f"- 발성 구간: **{hhmmss(voiced)}** ({len(segments)}개, 원본의 {voiced/duration*100:.0f}%)",
        f"- 대상자 구간: **{hhmmss(sum(s.dur for s in targets))}** ({len(targets)}개)",
        f"- 화자 판별: {mode}",
        "",
        "## 대상자 구간", "",
        "| # | 시각 | 길이 | 점수 | F0 | 검출 | 전사(보조) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in targets:
        lines.append(f"| {s.idx} | `{hhmmss(s.start)}` | {s.dur:.1f}s | {s.score:.2f} "
                     f"| {s.f0:.0f} | {s.src} | {s.text.replace('|', '/') or '—'} |")
    lines += ["", "검출 표시: `E` 에너지 VAD, `S` Silero VAD, `ES` 둘 다.", ""]
    dst.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(
        description="회기 영상의 발성 구간을 '누가·언제'로 인덱싱하고 리뷰어를 생성한다")
    ap.add_argument("video", type=Path)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--target", action="append", default=[],
                    help="대상자 실제 발화 구간 'MM:SS-MM:SS' (여러 번 지정 가능)")
    ap.add_argument("--labels", type=Path, default=None,
                    help="리뷰어에서 내보낸 labels.json — 참조 발화로 재사용")
    ap.add_argument("--speakers", type=int, default=2,
                    help="영상에 등장하는 화자 수(보호자가 함께면 3)")
    ap.add_argument("--vad", choices=["both", "energy", "silero"], default="both")
    ap.add_argument("--min-dur", type=float, default=0.12,
                    help="이보다 짧은 소리는 버린다. 단음절 발성까지 잡으려 낮게 잡았다")
    ap.add_argument("--max-gap", type=float, default=0.35)
    ap.add_argument("--sensitivity", type=float, default=0.25,
                    help="에너지 VAD 임계. 낮출수록 작은 소리까지 잡는다")
    ap.add_argument("--vad-threshold", type=float, default=0.3, help="Silero 임계")
    ap.add_argument("--join-gap", type=float, default=0.05,
                    help="두 VAD 결과를 합칠 때 이 간격 이내만 한 구간으로 본다")
    ap.add_argument("--pad", type=float, default=1.5, help="재생 시 앞뒤 여유(초)")
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--prompt", default=None, help="전사 유도용 초기 프롬프트(목표 어휘 등)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--vad-model", default=str(Path.home() / ".cache/whisper.cpp/ggml-silero-v5.1.2.bin"))
    ap.add_argument("--no-asr", action="store_true", help="전사 생략(빠름)")
    ap.add_argument("--proxy", action="store_true", help="브라우저용 저용량 사본 생성")
    ap.add_argument("--serve", action="store_true", help="리뷰어를 로컬 웹서버로 연다")
    args = ap.parse_args()

    video = args.video.expanduser().resolve()
    if not video.exists():
        print(f"영상을 찾을 수 없습니다: {video}", file=sys.stderr)
        return 1

    work = (args.workdir or video.parent / f"speech-index-{video.stem}").expanduser()
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "audio.wav"

    print("[1/6] 오디오 추출", flush=True)
    extract_audio(video, wav)
    x = load_wav(wav)
    duration = len(x) / SR

    print("[2/6] 발성 구간 검출", flush=True)
    e_spans = s_spans = []
    if args.vad in ("both", "energy"):
        e_spans = energy_vad(x, min_dur=args.min_dur, max_gap=args.max_gap,
                             sensitivity=args.sensitivity)
        print(f"      에너지 VAD: {len(e_spans)}개", flush=True)
    if args.vad in ("both", "silero"):
        s_spans = silero_vad(wav, Path(args.vad_model).expanduser(),
                             threshold=args.vad_threshold, min_dur=args.min_dur,
                             min_silence=args.max_gap, pad=0.1)
        print(f"      Silero VAD: {len(s_spans)}개"
              + ("  (모델 없음 — 건너뜀)" if not s_spans and args.vad == "both" else ""),
              flush=True)
    spans = merge_spans(e_spans, s_spans, join_gap=args.join_gap, min_dur=args.min_dur)
    if not spans:
        print("발성 구간이 없습니다. --sensitivity 를 낮춰보십시오.", file=sys.stderr)
        return 1
    print(f"      합집합: {len(spans)}개 / 총 {hhmmss(duration)}", flush=True)

    print("[3/6] 음향 임베딩", flush=True)
    segments = []
    for i, (a, b, src) in enumerate(spans, start=1):
        vec, rms_db, f0 = embed(x[int(a * SR):int(b * SR)])
        segments.append(Segment(idx=i, start=a, end=b, src=src, rms_db=rms_db, f0=f0,
                                embedding=normalize(vec).tolist()))
    E = np.array([s.embedding for s in segments])

    print("[4/6] 화자 판별", flush=True)
    refs: list[tuple[float, float]] = [parse_range(r) for r in args.target]
    if args.labels:
        payload = json.loads(Path(args.labels).expanduser().read_text(encoding="utf-8"))
        refs += [(t["start"], t["end"]) for t in payload.get("targets", [])]

    labels = kmeans(E, args.speakers)
    counts = np.bincount(labels, minlength=args.speakers)
    centroids = [normalize(E[labels == j].mean(axis=0)) if counts[j] else np.zeros(E.shape[1])
                 for j in range(args.speakers)]

    if refs:
        # 참조 발화는 '어느 무리가 대상자인가'를 고르는 데만 쓴다. 참조 한 조각의
        # 음소 특성에 직접 임계를 걸면 같은 화자의 다른 소리를 놓친다.
        ref_vec = normalize(np.mean(
            [normalize(embed(x[int(a * SR):int(b * SR)])[0]) for a, b in refs], axis=0))
        sims = [float(c @ ref_vec) for c in centroids]
        pick = int(np.argmax(sims))
        scores = E @ ref_vec
        for s, lab, sc in zip(segments, labels, scores):
            s.speaker, s.is_target, s.score = int(lab), bool(lab == pick), float(sc)
        mode = (f"참조 발화 {len(refs)}개 → {args.speakers}화자 무리 중 "
                f"{pick}번 무리({counts[pick]}구간)를 대상자로 확정 "
                f"(무리별 유사도 {', '.join(f'{v:.2f}' for v in sims)})")
        gap = sorted(sims)[-1] - sorted(sims)[-2] if len(sims) > 1 else 1.0
        if gap < 0.03:
            print(f"      ⚠ 화자 무리 간 음색 차이가 작습니다(유사도 차 {gap:.3f}). "
                  "참조 발화를 더 지정하거나 --speakers 를 조정하십시오.", flush=True)
    else:
        pick = int(np.argmin([c if c else 10**9 for c in counts]))
        for s, lab in zip(segments, labels):
            s.speaker, s.is_target = int(lab), bool(lab == pick)
            s.score = (1.0 if lab == pick else 0.0) + 0.5 * max(0.0, 1.0 - s.dur / 4.0)
        mode = (f"{args.speakers}화자 무리로 분리 — 가장 적게 말한 "
                f"{pick}번 무리({counts[pick]}/{len(segments)}구간)를 대상자로 추정")

    n_target = sum(s.is_target for s in segments)
    print(f"      대상자 후보 {n_target}개 / 전체 {len(segments)}개", flush=True)

    asr_json = work / "asr.json"
    if not args.no_asr:
        binary, model = find_whisper(), default_model(args.model)
        if binary and model:
            print(f"[5/6] 전사 (보조, {model.name})", flush=True)
            transcribe(wav, asr_json, model, binary, args.lang, args.prompt)
        else:
            print(f"[5/6] 전사 건너뜀 ({'모델' if binary else 'whisper.cpp'} 없음)", flush=True)
    else:
        print("[5/6] 전사 건너뜀 (--no-asr)", flush=True)
    attach_text(segments, load_asr(asr_json))

    print("[6/6] 리뷰어·인덱스 생성", flush=True)
    if args.proxy:
        proxy = work / "proxy.mp4"
        print("      프록시 영상 인코딩", flush=True)
        make_proxy(video, proxy)
        video_src = proxy.name
    else:
        link = work / f"source{video.suffix}"
        if not link.exists():
            link.symlink_to(video)
        video_src = link.name

    write_reviewer(work / "reviewer.html", video_src, video.stem, duration, segments, args.pad)
    write_index(work / "index.md", video.stem, duration, segments, mode)
    write_vtt(work / "subtitles.vtt", segments)
    (work / "segments.json").write_text(json.dumps(
        [{**{k: v for k, v in s.__dict__.items() if k != "embedding"},
          "dur": round(s.dur, 2)} for s in segments],
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n완료 → {work}")
    print("  reviewer.html   영상 + 타임라인 + 구간 점프 (주 산출물)")
    print("  index.md        대상자 구간 타임코드 표")
    print("  subtitles.vtt   기존 플레이어용 화자 표시 자막")
    print("  segments.json   전체 구간 원자료")

    if args.serve:
        import http.server, socketserver, threading, webbrowser, functools
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(work))
        with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
            port = httpd.server_address[1]
            url = f"http://127.0.0.1:{port}/reviewer.html"
            print(f"\n리뷰어: {url}   (Ctrl+C 로 종료)")
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n서버를 닫았습니다.")
    else:
        print(f"\n리뷰어 열기:  open '{work / 'reviewer.html'}'")
        print("  (재생이 안 되면 --serve 또는 --proxy 를 붙여 다시 실행하십시오)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
