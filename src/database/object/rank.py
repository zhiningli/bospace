from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Rank:
    rank_id: int | None = None
    model_idx: int = 0
    dataset_idx: int = 0
    result: list[dict[str, float|int]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "rank_id": self.rank_id,
            "model_idx": self.model_idx,
            "dataset_idx": self.dataset_idx,
            "results": self.results,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_row(cls, row: tuple):
        return cls(
            rank_id=row[0],
            model_idx=row[1],
            dataset_idx=row[2],
            results=row[3],
            created_at=row[4],
        )