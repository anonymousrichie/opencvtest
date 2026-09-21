"""Local enrollment database: persists named face embeddings + profile info to disk.

A pickle of {name: PersonRecord} is sufficient for a base system -- callers
only ever go through add()/best_match()/get_info()/names(), so swapping the
backend for SQLite or a vector index later is a drop-in change that touches
no other file.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PersonInfo:
    """Optional profile fields captured alongside a face enrollment.

    All fields are free-text and optional -- blank means "not provided",
    not "unknown value". Kept as strings rather than typed (e.g. age as int)
    since this is operator-entered data with no downstream computation on it.
    """

    age: str = ""
    department: str = ""
    person_id: str = ""
    nickname: str = ""
    hall: str = ""


@dataclass
class PersonRecord:
    name: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    info: PersonInfo = field(default_factory=PersonInfo)


class EnrollmentDatabase:
    """Stores one or more face embeddings plus profile info per enrolled name."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._entries: dict[str, PersonRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._db_path.exists():
            return
        with open(self._db_path, "rb") as f:
            raw: dict[str, object] = pickle.load(f)
        # Migrate the pre-profile-info format ({name: [embeddings...]}) some
        # existing databases were saved in, so upgrading this code doesn't
        # strand already-enrolled faces behind a pickle-format mismatch.
        self._entries = {
            name: value if isinstance(value, PersonRecord) else PersonRecord(name=name, embeddings=list(value))
            for name, value in raw.items()
        }

    def save(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._db_path, "wb") as f:
            pickle.dump(self._entries, f)

    def add(self, name: str, embedding: np.ndarray, info: PersonInfo | None = None) -> None:
        """Enroll a new embedding under `name` (multiple embeddings per name are
        fine and improve matching robustness across lighting/pose).

        `info` is merged onto any existing profile field-by-field -- a blank
        field in `info` leaves the previously stored value untouched, so
        re-enrolling from a new angle doesn't require retyping the whole
        profile each time.
        """
        record = self._entries.setdefault(name, PersonRecord(name=name))
        record.embeddings.append(embedding.astype(np.float32))
        if info is not None:
            record.info = _merge_info(record.info, info)
        self.save()

    def best_match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return (name, cosine_similarity) of the closest enrolled embedding,
        or (None, -1.0) if the database is empty. Thresholding against
        "unknown" is the caller's responsibility."""
        best_name: str | None = None
        best_score = -1.0
        for name, record in self._entries.items():
            for ref in record.embeddings:
                score = _cosine_similarity(embedding, ref)
                if score > best_score:
                    best_name, best_score = name, score
        return best_name, best_score

    def get_info(self, name: str) -> PersonInfo | None:
        """Return the stored profile for `name`, or None if never set."""
        record = self._entries.get(name)
        return record.info if record else None

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def is_empty(self) -> bool:
        return not self._entries


def _merge_info(old: PersonInfo, new: PersonInfo) -> PersonInfo:
    return PersonInfo(
        age=new.age or old.age,
        department=new.department or old.department,
        person_id=new.person_id or old.person_id,
        nickname=new.nickname or old.nickname,
        hall=new.hall or old.hall,
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
