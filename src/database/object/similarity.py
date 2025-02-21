from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Similarity:
    similarity_idx: int | None = None
    object_type: str = "" #Enum either model or dataset
    object_1_idx: int | None = None  # Foreign key for dataset_idx or model_idx
    object_1: list[float] = field(default_factory=list)

    object_2_idx: int | None = None  # Foreign key for dataset_idx or model_idx
    object_2: list[float] = field(default_factory=list)

    similarity: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert the Similarity object to a dictionary."""
        return {
            "similarity_idx": self.rank_idx,
            "object_type": self.object_1_type,
            "object_1_idx": self.object_1_idx,
            "object_1": self.object_1,
            "object_2_idx": self.object_2_idx,
            "object_2": self.object_2,
            "similarity": self.rank,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_row(cls, row: tuple):
        """Create a Rank object from a database row."""
        return cls(
            similarity_idx=row[0],
            object_type=row[1],
            object_1_idx=row[2],
            object_1=row[3],
            object_2_idx=row[4],
            object_2=row[5],
            similarity=row[6],
            created_at=row[7],
        )
