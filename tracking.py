"""Lightweight IoU-based multi-object tracker for stable face track IDs.

A full tracker (SORT/DeepSORT/ByteTrack) is overkill for a handful of faces in
frame: greedy IoU matching between consecutive frames' bounding boxes is
dependency-free, easy to reason about, and plenty stable once detections are
already smooth frame-to-frame (which YuNet's are). This is the extension point
if the base system later needs to survive fast motion or long occlusions.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import TrackingConfig

BBox = tuple[int, int, int, int]  # x, y, w, h


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """A tracked face, persisted across frames under a stable `track_id`."""

    track_id: int
    bbox: BBox
    missed_frames: int = 0
    identity: str = "Unknown"
    confidence: float = 0.0
    # Large default so a brand-new track is always recognized on its first frame.
    frames_since_recognition: int = 10_000
    emotion: str = "Neutral"
    emotion_score: float = 0.0
    # Large default so a brand-new track is always classified on its first frame.
    frames_since_emotion: int = 10_000


class IoUTracker:
    """Assigns persistent integer IDs to detections via greedy IoU matching."""

    def __init__(self, cfg: TrackingConfig) -> None:
        self._cfg = cfg
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list[BBox]) -> list[Track]:
        """Match `detections` against existing tracks and return one Track per
        detection, in the same order as `detections`. New detections spawn new
        tracks; unmatched existing tracks age out after `max_missed_frames`.
        """
        unmatched_track_ids = set(self._tracks.keys())
        unmatched_det_idxs = set(range(len(detections)))

        candidate_pairs: list[tuple[float, int, int]] = []
        for tid, track in self._tracks.items():
            for di, det in enumerate(detections):
                iou = _iou(track.bbox, det)
                if iou >= self._cfg.iou_match_threshold:
                    candidate_pairs.append((iou, tid, di))
        candidate_pairs.sort(key=lambda p: p[0], reverse=True)

        det_to_track: dict[int, int] = {}
        for _iou_score, tid, di in candidate_pairs:
            if tid in unmatched_track_ids and di in unmatched_det_idxs:
                det_to_track[di] = tid
                unmatched_track_ids.discard(tid)
                unmatched_det_idxs.discard(di)

        for di, tid in det_to_track.items():
            track = self._tracks[tid]
            track.bbox = detections[di]
            track.missed_frames = 0

        for tid in unmatched_track_ids:
            self._tracks[tid].missed_frames += 1

        for di in sorted(unmatched_det_idxs):
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(track_id=tid, bbox=detections[di])
            det_to_track[di] = tid

        self._tracks = {
            tid: t for tid, t in self._tracks.items() if t.missed_frames <= self._cfg.max_missed_frames
        }

        return [self._tracks[det_to_track[di]] for di in range(len(detections))]
