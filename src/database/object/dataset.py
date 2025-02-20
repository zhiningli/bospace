from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Dataset:
    """Dataclass representing a row in the 'dataset' table"""

    dataset_idx: int | None = None
    code: str = ""
    meta_features: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "dataset_idx": self.dataset_idx,
            "code": self.code,
            "meta_features": self.meta_features,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_row(cls, row: tuple):
        return cls(
            dataset_idx = row[0],
            code = row[1],
            feature_vector = row[2],
            created_At = row[3],
        )