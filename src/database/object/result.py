from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

@dataclass
class Result:
    """Dataclass representing a row in the 'results' table."""

    result_id: int | None = None
    script_idx: int = 0
    model_idx: int = 0
    dataset_idx: int = 0
    result_type: str = ""  # ENUM type: 'unconstrained', 'constrained', etc.
    sgd_best_performing_configuration: Dict[str, int | float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert the Result object to a dictionary for easier serialization."""
        return {
            "result_id": self.result_id,
            "script_idx": self.script_idx,
            "model_idx": self.model_idx,
            "dataset_idx": self.dataset_idx,
            "result_type": self.result_type,
            "sgd_best_performing_configuration": self.sgd_best_performing_configuration,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_row(cls, row: tuple):
        """Create a Result instance from a database row."""
        return cls(
            result_id=row[0],
            script_idx=row[1],
            model_idx=row[2],
            dataset_idx=row[3],
            result_type=row[4],
            sgd_best_performing_configuration=row[5],
            created_at=row[6],
        )
