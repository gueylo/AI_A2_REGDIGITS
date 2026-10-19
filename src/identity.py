"""Student identity and deterministic seed validation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Identity:
    student_id: str
    registration_digits: str
    last_four: str
    seed: int
    verification_code: str

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


def identity_from_student_id(student_id: str, seed: int) -> Identity:
    """Validate an institutional username and its last-four-digit seed."""
    value = student_id.strip()
    if not value or "@" not in value:
        raise ValueError("--student-id must be a full institutional email/username")
    local, domain = value.rsplit("@", 1)
    if not local or not domain:
        raise ValueError("--student-id must contain a non-empty local part and domain")
    groups = re.findall(r"\d+", local)
    if not groups:
        raise ValueError("the student-id local part must contain registration digits")
    registration_digits = "".join(groups)
    if len(registration_digits) < 4:
        raise ValueError("student-id must contain at least four registration digits")
    last_four = registration_digits[-4:]
    expected = int(last_four)
    if seed != expected:
        raise ValueError(
            f"seed mismatch: expected int(last four registration digits) = {expected}"
        )
    return Identity(
        student_id=value,
        registration_digits=registration_digits,
        last_four=last_four,
        seed=seed,
        verification_code=f"AI-A2-{last_four}",
    )

