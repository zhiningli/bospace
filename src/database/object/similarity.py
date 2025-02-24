from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Similarity:
    similarity_idx: int | None = None
    object_1_idx: int | None = None  # Foreign key for dataset_idx or model_idx
    object_1_feature: list[float] = field(default_factory=list)

    object_2_idx: int | None = None  # Foreign key for dataset_idx or model_idx
    object_2_feature: list[float] = field(default_factory=list)

    similarity: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert the Similarity object to a dictionary."""
        return {
            "similarity_idx": self.rank_idx,
            "object_1_idx": self.object_1_idx,
            "object_1_feature": self.object_1,
            "object_2_idx": self.object_2_idx,
            "object_2_feature": self.object_2,
            "similarity": self.rank,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_row(cls, row: tuple):
        """Create a Rank object from a database row."""
        return cls(
            similarity_idx=row[0],
            object_1_idx=row[1],
            object_1_feature=row[2],
            object_2_idx=row[3],
            object_2_feature=row[4],
            similarity=row[5],
            created_at=row[6],
        )
