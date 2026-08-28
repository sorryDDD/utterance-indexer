#!/usr/bin/env python3
"""저장소 밖의 정답 라벨과 segments.json을 로컬에서 채점한다.

실제 회기 파일은 저장소에 넣지 않는다. 이 스크립트는 경로나 라벨 원문을
결과 표에 싣지 않고 집계값만 출력한다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OVERLAP_FRACTION = 0.15
ALLOWED_CATEGORIES = {"target", "clinician", "unsure", "other_voice"}
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ValidationError(ValueError):
    """입력 형식이 채점 계약에 맞지 않는다."""


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Label:
    category: str
    start: float
    end: float
    covered_parts: tuple[Interval, ...]

    @property
    def scored_duration(self) -> float:
        return sum(part.duration for part in self.covered_parts)


@dataclass(frozen=True)
class Segment:
    idx: int
    start: float
    end: float
    is_target: bool
    speaker: int
    src: str
    f0: float
    covered_parts: tuple[Interval, ...]

    @property
    def scored_duration(self) -> float:
        return sum(part.duration for part in self.covered_parts)


@dataclass(frozen=True)
class SegmentSet:
    scored: tuple[Segment, ...]
    global_clusters: tuple[dict[str, Any], ...]


def finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field}: 유한한 수가 아닙니다")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{field}: 유한한 수가 아닙니다")
    return result


def interval_overlap(a: Interval, b: Interval) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def normalize_intervals(items: list[Interval]) -> tuple[Interval, ...]:
    if not items:
        return ()
    result: list[Interval] = []
    for item in sorted(items, key=lambda value: (value.start, value.end)):
        if not result or item.start > result[-1].end:
            result.append(item)
        else:
            result[-1] = Interval(result[-1].start, max(result[-1].end, item.end))
    return tuple(result)


def clip_interval(start: float, end: float,
                  covered: tuple[Interval, ...]) -> tuple[Interval, ...]:
    parts = []
    for scope in covered:
        left, right = max(start, scope.start), min(end, scope.end)
        if right > left:
            parts.append(Interval(left, right))
    return tuple(parts)


def union_duration(parts: list[Interval]) -> float:
    return sum(item.duration for item in normalize_intervals(parts))


def label_segment_overlap(label: Label, segment: Segment) -> float:
    return sum(interval_overlap(a, b)
               for a in label.covered_parts for b in segment.covered_parts)


def labels_overlap(a: Label, b: Label) -> float:
    return sum(interval_overlap(x, y)
               for x in a.covered_parts for y in b.covered_parts)


def is_hit(label: Label, segment: Segment) -> bool:
    return label_segment_overlap(label, segment) > OVERLAP_FRACTION * label.scored_duration


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("입력 JSON을 읽지 못했습니다") from exc


def load_labels(path: Path) -> tuple[dict[str, Any], list[Label], tuple[Interval, ...]]:
    payload = load_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValidationError("라벨 schema_version은 2여야 합니다")
    duration = finite_number(payload.get("duration"), "duration")
    if duration <= 0:
        raise ValidationError("duration은 0보다 커야 합니다")

    raw_covered = payload.get("covered")
    if not isinstance(raw_covered, list) or not raw_covered:
        raise ValidationError("covered가 비어 있습니다")
    covered_items = []
    for i, item in enumerate(raw_covered):
        if not isinstance(item, dict):
            raise ValidationError(f"covered[{i}]가 객체가 아닙니다")
        start = finite_number(item.get("start"), f"covered[{i}].start")
        end = finite_number(item.get("end"), f"covered[{i}].end")
        if start < 0 or end <= start or end > duration:
            raise ValidationError(f"covered[{i}]의 시각 범위가 잘못되었습니다")
        covered_items.append(Interval(start, end))
    covered = normalize_intervals(covered_items)

    raw_labels = payload.get("labels")
    if not isinstance(raw_labels, list):
        raise ValidationError("labels가 배열이 아닙니다")
    result: list[Label] = []
    previous_start = -math.inf
    for i, item in enumerate(raw_labels):
        if not isinstance(item, dict):
            raise ValidationError(f"labels[{i}]가 객체가 아닙니다")
        category = item.get("category")
        if category not in ALLOWED_CATEGORIES:
            raise ValidationError(f"labels[{i}].category가 허용 범위 밖입니다")
        start = finite_number(item.get("start"), f"labels[{i}].start")
        end = finite_number(item.get("end"), f"labels[{i}].end")
        if start < previous_start:
            raise ValidationError("labels가 시작 시각 오름차순이 아닙니다")
        if start < 0 or end <= start or end > duration:
            raise ValidationError(f"labels[{i}]의 시각 범위가 잘못되었습니다")
        previous_start = start
        parts = clip_interval(start, end, covered)
        if parts:
            result.append(Label(category, start, end, parts))
    return payload, result, covered


def load_segments(path: Path, covered: tuple[Interval, ...],
                  duration: float) -> SegmentSet:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValidationError("segments.json 최상위 값이 배열이 아닙니다")
    if not payload:
        raise ValidationError("segments.json이 비어 있습니다")
    result = []
    global_clusters: dict[int, dict[str, Any]] = {}
    previous_start = -math.inf
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValidationError(f"segments[{i}]가 객체가 아닙니다")
        start = finite_number(item.get("start"), f"segments[{i}].start")
        end = finite_number(item.get("end"), f"segments[{i}].end")
        f0 = finite_number(item.get("f0"), f"segments[{i}].f0")
        if start < previous_start:
            raise ValidationError("segments가 시작 시각 오름차순이 아닙니다")
        if start < 0 or end <= start or end > duration + 0.1:
            raise ValidationError(f"segments[{i}]의 시각 범위가 잘못되었습니다")
        previous_start = start
        is_target = item.get("is_target")
        speaker = item.get("speaker")
        src = item.get("src")
        if not isinstance(is_target, bool):
            raise ValidationError(f"segments[{i}].is_target이 bool이 아닙니다")
        if isinstance(speaker, bool) or not isinstance(speaker, int):
            raise ValidationError(f"segments[{i}].speaker가 정수가 아닙니다")
        if speaker < 0:
            raise ValidationError(f"segments[{i}].speaker가 음수입니다")
        if src not in {"E", "S", "ES"}:
            raise ValidationError(f"segments[{i}].src가 허용 범위 밖입니다")
        raw_idx = item.get("idx", i + 1)
        if isinstance(raw_idx, bool) or not isinstance(raw_idx, int):
            raise ValidationError(f"segments[{i}].idx가 정수가 아닙니다")
        cluster = global_clusters.setdefault(
            speaker, {"speaker": speaker, "segment_count": 0,
                      "seconds": 0.0, "is_current_target": False})
        cluster["segment_count"] += 1
        cluster["seconds"] += end - start
        cluster["is_current_target"] |= is_target
        parts = clip_interval(start, end, covered)
        if parts:
            result.append(Segment(raw_idx, start, end,
                                  is_target, speaker, src, f0, parts))
    ordered_clusters = tuple(global_clusters[key] for key in sorted(global_clusters))
    return SegmentSet(tuple(result), ordered_clusters)


def matching_segments(label: Label, segments: list[Segment]) -> list[Segment]:
    return [segment for segment in segments if is_hit(label, segment)]


def recall(labels: list[Label], segments: list[Segment]) -> dict[str, Any]:
    detected = sum(bool(matching_segments(label, segments)) for label in labels)
    return {"total": len(labels), "detected": detected,
            "recall": rate(detected, len(labels))}


def speaker_metrics(targets: list[Label], clinicians: list[Label],
                    unsures: list[Label], segments: list[Segment]) -> dict[str, Any]:
    eligible_targets = [label for label in targets
                        if not any(labels_overlap(label, clinician) > 0
                                   for clinician in clinicians)]
    target_detected = target_success = 0
    for label in eligible_targets:
        hits = matching_segments(label, segments)
        target_detected += bool(hits)
        target_success += any(segment.is_target for segment in hits)

    eligible_clinicians = [label for label in clinicians
                           if not any(labels_overlap(label, target) > 0
                                      for target in targets)
                           and not any(labels_overlap(label, unsure) > 0
                                       for unsure in unsures)]
    clinician_detected = clinician_leaked = 0
    for label in eligible_clinicians:
        hits = matching_segments(label, segments)
        clinician_detected += bool(hits)
        clinician_leaked += any(segment.is_target for segment in hits)

    return {
        "target_total": len(eligible_targets),
        "target_detected": target_detected,
        "target_candidate_success": target_success,
        "target_candidate_rate": rate(target_success, len(eligible_targets)),
        "conditional_assignment_rate": rate(target_success, target_detected),
        "clinician_total": len(eligible_clinicians),
        "clinician_detected": clinician_detected,
        "clinician_leaked": clinician_leaked,
        "clinician_leak_rate": rate(clinician_leaked, len(eligible_clinicians)),
        "conditional_clinician_leak_rate": rate(clinician_leaked, clinician_detected),
    }


def false_positive_metrics(labels: list[Label], segments: list[Segment]) -> dict[str, Any]:
    false_segments = [segment for segment in segments
                      if not any(label_segment_overlap(label, segment) > 0
                                 for label in labels)]
    false_parts = [part for segment in false_segments for part in segment.covered_parts]
    all_parts = [part for segment in segments for part in segment.covered_parts]
    return {
        "segment_count": len(false_segments),
        "seconds": union_duration(false_parts),
        "count_ratio": rate(len(false_segments), len(segments)),
        "time_ratio": (union_duration(false_parts) / union_duration(all_parts)
                       if all_parts and union_duration(all_parts) else None),
    }


def flag_metrics(name: str, flagged: list[Segment], labels: list[Label],
                 targets: list[Label], segments: list[Segment],
                 baseline_recall: float | None) -> dict[str, Any]:
    kept = [segment for segment in segments if segment not in flagged]
    kept_recall = recall(targets, kept)["recall"]
    false_flagged = [segment for segment in flagged
                     if not any(label_segment_overlap(label, segment) > 0
                                for label in labels)]
    flagged_seconds = union_duration(
        [part for segment in flagged for part in segment.covered_parts])
    false_seconds = union_duration(
        [part for segment in false_flagged for part in segment.covered_parts])
    return {
        "name": name,
        "segment_count": len(flagged),
        "seconds": flagged_seconds,
        "false_segment_count": len(false_flagged),
        "false_segment_ratio": rate(len(false_flagged), len(flagged)),
        "false_seconds": false_seconds,
        "false_time_ratio": false_seconds / flagged_seconds if flagged_seconds else None,
        "target_recall_after_rejection": kept_recall,
        "target_recall_drop": (baseline_recall - kept_recall
                               if baseline_recall is not None and kept_recall is not None
                               else None),
    }


def cluster_metrics(targets: list[Label], clinicians: list[Label],
                    unsures: list[Label], segments: list[Segment],
                    global_clusters: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    eligible = [label for label in targets
                if not any(labels_overlap(label, clinician) > 0
                           for clinician in clinicians)]
    eligible_clinicians = [label for label in clinicians
                           if not any(labels_overlap(label, target) > 0
                                      for target in targets)
                           and not any(labels_overlap(label, unsure) > 0
                                       for unsure in unsures)]
    rows = []
    for speaker in sorted({segment.speaker for segment in segments}):
        members = [segment for segment in segments if segment.speaker == speaker]
        success = sum(any(is_hit(label, segment) for segment in members)
                      for label in eligible)
        clinician_leak = sum(any(is_hit(label, segment) for segment in members)
                             for label in eligible_clinicians)
        rows.append({
            "speaker": speaker,
            "segment_count": len(members),
            "seconds": union_duration(
                [part for segment in members for part in segment.covered_parts]),
            "target_success": success,
            "target_success_rate": rate(success, len(eligible)),
            "clinician_leak": clinician_leak,
            "clinician_leak_rate": rate(clinician_leak, len(eligible_clinicians)),
            "is_current_target": any(segment.is_target for segment in members),
        })
    best = max((row["target_success_rate"] for row in rows
                if row["target_success_rate"] is not None), default=None)
    count_pick = min(global_clusters, key=lambda row: row["segment_count"])["speaker"]
    duration_pick = min(global_clusters, key=lambda row: row["seconds"])["speaker"]
    return {"covered_clusters": rows, "global_clusters": global_clusters,
            "global_count_pick": count_pick, "global_duration_pick": duration_pick,
            "diagnostic_upper_bound": best}


def evaluate_run(labels: list[Label], segment_set: SegmentSet) -> dict[str, Any]:
    segments = list(segment_set.scored)
    targets = [label for label in labels if label.category == "target"]
    clinicians = [label for label in labels if label.category == "clinician"]
    unsures = [label for label in labels if label.category == "unsure"]
    target_result = recall(targets, segments)
    participant_result = recall(targets + unsures, segments)
    short_targets = [label for label in targets if label.end - label.start < 0.5]
    short_speaker = speaker_metrics(short_targets, clinicians, [], segments)
    flags = [
        flag_metrics("energy_only", [s for s in segments if s.src == "E"],
                     labels, targets, segments, target_result["recall"]),
        flag_metrics("f0_zero", [s for s in segments if s.f0 == 0.0],
                     labels, targets, segments, target_result["recall"]),
    ]
    return {
        "scored_segment_count": len(segments),
        "scored_segment_seconds": union_duration(
            [part for segment in segments for part in segment.covered_parts]),
        "target_detection": target_result,
        "participant_detection": participant_result,
        "short_target": {
            **recall(short_targets, segments),
            "speaker_eligible": short_speaker["target_total"],
            "target_candidate_success": short_speaker["target_candidate_success"],
            "target_candidate_rate": short_speaker["target_candidate_rate"],
        },
        "false_positive": false_positive_metrics(labels, segments),
        "speaker": speaker_metrics(targets, clinicians, unsures, segments),
        "flags": {item["name"]: {k: v for k, v in item.items() if k != "name"}
                  for item in flags},
        "cluster_diagnostic": cluster_metrics(
            targets, clinicians, unsures, segments, segment_set.global_clusters),
    }


def paired_detection(targets: list[Label], a: list[Segment],
                     b: list[Segment]) -> dict[str, int]:
    cells = {"both": 0, "first_only": 0, "second_only": 0, "neither": 0}
    for label in targets:
        hit_a = bool(matching_segments(label, a))
        hit_b = bool(matching_segments(label, b))
        key = ("both" if hit_a and hit_b else "first_only" if hit_a
               else "second_only" if hit_b else "neither")
        cells[key] += 1
    return cells


def label_summary(labels: list[Label], covered: tuple[Interval, ...]) -> dict[str, Any]:
    counts = {category: sum(label.category == category for label in labels)
              for category in sorted(ALLOWED_CATEGORIES)}
    targets = [label for label in labels if label.category == "target"]
    clinicians = [label for label in labels if label.category == "clinician"]
    return {
        "covered_seconds": sum(item.duration for item in covered),
        "counts": counts,
        "target_under_0_5": sum(label.end - label.start < 0.5 for label in targets),
        "target_overlapping_clinician": sum(
            any(labels_overlap(label, clinician) > 0 for clinician in clinicians)
            for label in targets),
    }


def evaluate(label_path: Path, segment_specs: list[tuple[str, Path]]) -> dict[str, Any]:
    label_payload, labels, covered = load_labels(label_path)
    duration = finite_number(label_payload["duration"], "duration")
    runs = {name: load_segments(path, covered, duration)
            for name, path in segment_specs}
    targets = [label for label in labels if label.category == "target"]
    paired = {}
    for (name_a, segments_a), (name_b, segments_b) in itertools.combinations(runs.items(), 2):
        paired[f"{name_a}__vs__{name_b}"] = {
            "first": name_a,
            "second": name_b,
            **paired_detection(targets, list(segments_a.scored), list(segments_b.scored)),
        }
    return {
        "schema_version": 1,
        "overlap_fraction": OVERLAP_FRACTION,
        "labels": label_summary(labels, covered),
        "runs": {name: evaluate_run(labels, segments) for name, segments in runs.items()},
        "paired_target_detection": paired,
    }


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def print_report(report: dict[str, Any]) -> None:
    summary = report["labels"]
    counts = summary["counts"]
    print("라벨 집계")
    print(f"  들은 범위 {summary['covered_seconds']:.2f}초 · 대상자 {counts['target']} · "
          f"임상가 {counts['clinician']} · 보류 {counts['unsure']} · "
          f"제3자 {counts['other_voice']}")
    print(f"  0.5초 미만 대상자 {summary['target_under_0_5']} · "
          f"임상가 중첩 대상자 {summary['target_overlapping_clinician']}")
    for name, run in report["runs"].items():
        target = run["target_detection"]
        speaker = run["speaker"]
        false = run["false_positive"]
        short = run["short_target"]
        print(f"\n[{name}]")
        print(f"  대상자 검출 {target['detected']}/{target['total']} "
              f"({pct(target['recall'])})")
        print(f"  화자 후보 {speaker['target_candidate_success']}/{speaker['target_total']} "
              f"({pct(speaker['target_candidate_rate'])}) · 검출분 조건부 "
              f"{pct(speaker['conditional_assignment_rate'])}")
        print(f"  0.5초 미만 검출 {short['detected']}/{short['total']} "
              f"({pct(short['recall'])}) · 후보 {pct(short['target_candidate_rate'])}")
        print(f"  오탐 총량 {false['segment_count']}개 · {false['seconds']:.2f}초")
        for flag_name, flag in run["flags"].items():
            print(f"  {flag_name}: {flag['segment_count']}개 · 비참여자 "
                  f"{pct(flag['false_segment_ratio'])} · 대상자 재현율 하락 "
                  f"{pct(flag['target_recall_drop'])}")
    for comparison in report["paired_target_detection"].values():
        print(f"\n[{comparison['first']} ↔ {comparison['second']}] 대상자 짝비교")
        print(f"  둘 다 {comparison['both']} · 앞만 {comparison['first_only']} · "
              f"뒤만 {comparison['second_only']} · 둘 다 놓침 {comparison['neither']}")


def parse_segment_specs(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValidationError("--segments는 이름=경로 형식이어야 합니다")
        name, raw_path = value.split("=", 1)
        if not NAME_RE.fullmatch(name) or name in seen or not raw_path:
            raise ValidationError("--segments 이름이 잘못되었거나 중복되었습니다")
        seen.add(name)
        result.append((name, Path(raw_path).expanduser()))
    return result


def assert_close(actual: float | None, expected: float, tolerance: float = 1e-9) -> None:
    assert actual is not None and abs(actual - expected) <= tolerance, (actual, expected)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="utterance-eval-test-") as tmp:
        root = Path(tmp)
        labels = {
            "schema_version": 2,
            "duration": 30.0,
            "covered": [{"start": 20.0, "end": 30.0}, {"start": 0.0, "end": 10.0}],
            "labels": [
                {"start": 1.0, "end": 2.0, "category": "target"},
                {"start": 3.0, "end": 4.0, "category": "target"},
                {"start": 5.0, "end": 5.4, "category": "target"},
                {"start": 7.0, "end": 8.0, "category": "unsure"},
                {"start": 8.0, "end": 9.0, "category": "clinician"},
                {"start": 21.0, "end": 22.0, "category": "target"},
                {"start": 21.5, "end": 21.8, "category": "clinician"},
                {"start": 24.0, "end": 25.0, "category": "other_voice"},
            ],
        }
        both = [
            {"idx": 1, "start": 0.1, "end": 0.5, "is_target": False,
             "speaker": 1, "src": "E", "f0": 0.0},
            {"idx": 2, "start": 0.3, "end": 0.7, "is_target": False,
             "speaker": 1, "src": "E", "f0": 0.0},
            {"idx": 3, "start": 1.1, "end": 1.4, "is_target": True,
             "speaker": 0, "src": "ES", "f0": 250.0},
            {"idx": 4, "start": 3.0, "end": 3.15, "is_target": True,
             "speaker": 0, "src": "S", "f0": 250.0},
            {"idx": 5, "start": 5.1, "end": 5.2, "is_target": False,
             "speaker": 1, "src": "S", "f0": 0.0},
            {"idx": 6, "start": 7.1, "end": 7.4, "is_target": False,
             "speaker": 1, "src": "S", "f0": 180.0},
            {"idx": 7, "start": 8.1, "end": 8.4, "is_target": True,
             "speaker": 0, "src": "S", "f0": 180.0},
            {"idx": 8, "start": 10.0, "end": 20.0, "is_target": True,
             "speaker": 0, "src": "E", "f0": 0.0},
            {"idx": 9, "start": 21.0, "end": 22.0, "is_target": True,
             "speaker": 0, "src": "ES", "f0": 300.0},
            {"idx": 10, "start": 24.1, "end": 24.2, "is_target": False,
             "speaker": 1, "src": "S", "f0": 210.0},
        ]
        silero = [segment for segment in both if segment["src"] != "E" and segment["idx"] != 5]
        label_path = root / "labels.json"
        both_path = root / "both.json"
        silero_path = root / "silero.json"
        label_path.write_text(json.dumps(labels), encoding="utf-8")
        both_path.write_text(json.dumps(both), encoding="utf-8")
        silero_path.write_text(json.dumps(silero), encoding="utf-8")

        report = evaluate(label_path, [("both", both_path), ("silero", silero_path)])
        assert report["labels"]["target_under_0_5"] == 1
        assert report["labels"]["target_overlapping_clinician"] == 1
        run = report["runs"]["both"]
        assert run["scored_segment_count"] == 9
        assert run["target_detection"]["detected"] == 3
        assert run["target_detection"]["total"] == 4
        assert run["participant_detection"]["detected"] == 4
        assert run["short_target"]["detected"] == 1
        assert run["speaker"]["target_total"] == 3
        assert run["speaker"]["target_candidate_success"] == 1
        assert run["speaker"]["clinician_leaked"] == 1
        assert run["false_positive"]["segment_count"] == 2
        assert_close(run["false_positive"]["seconds"], 0.6)
        assert_close(run["flags"]["energy_only"]["target_recall_drop"], 0.0)
        assert run["flags"]["f0_zero"]["target_recall_drop"] > 0
        clusters = run["cluster_diagnostic"]
        assert clusters["global_count_pick"] == 0
        assert clusters["global_duration_pick"] == 1
        pair = report["paired_target_detection"]["both__vs__silero"]
        assert pair == {"first": "both", "second": "silero",
                        "both": 2, "first_only": 1, "second_only": 0, "neither": 1}

        exact_label = Label("target", 100.0, 200.0, (Interval(100.0, 200.0),))
        exact_segment = Segment(1, 100.0, 115.0, True, 0, "S", 200.0,
                                (Interval(100.0, 115.0),))
        assert not is_hit(exact_label, exact_segment)

        def expect_invalid(payload: dict[str, Any], message: str) -> None:
            broken_path = root / "broken.json"
            broken_path.write_text(json.dumps(payload), encoding="utf-8")
            try:
                evaluate(broken_path, [("both", both_path)])
            except ValidationError:
                return
            raise AssertionError(message)

        broken_schema = json.loads(json.dumps(labels))
        broken_schema["schema_version"] = 1
        expect_invalid(broken_schema, "잘못된 스키마가 거부되지 않았습니다")
        broken_category = json.loads(json.dumps(labels))
        broken_category["labels"][0]["category"] = "noise"
        expect_invalid(broken_category, "잘못된 범주가 거부되지 않았습니다")
        broken_time = json.loads(json.dumps(labels))
        broken_time["labels"][0]["end"] = broken_time["labels"][0]["start"]
        expect_invalid(broken_time, "잘못된 시각이 거부되지 않았습니다")
    print("채점 하니스 자체 검증 통과 — 10개 범주")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="정답 라벨과 segments.json을 로컬에서 채점합니다")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--segments", action="append", default=[])
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    try:
        if args.self_test:
            self_test()
            return 0
        if args.labels is None or not args.segments:
            parser.error("--labels와 하나 이상의 --segments가 필요합니다")
        specs = parse_segment_specs(args.segments)
        report = evaluate(args.labels.expanduser(), specs)
        print_report(report)
        if args.json_output:
            args.json_output.expanduser().write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    except ValidationError as exc:
        print(f"채점 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
