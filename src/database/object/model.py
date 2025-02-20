from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Model:
    """Dataclass representing a row in the 'models' table """
    model_idx: int | None = None
    code: dict[str, str] = field(default_factory=dict)
    feature_vector: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "model_idx": self.model_idx,
            "code": self.code,
            "feature_vector": self.feature_vector,
            "created_at": self.created_at.isoformat(),
        }
    

    @classmethod
    def from_row(cls, row: tuple):
        """Convert a database row to a model instance"""
        return cls(
            model_idx = row[0],
            code = row[1],
            feature_vector = row[2],
            created_at = row[3],
        )