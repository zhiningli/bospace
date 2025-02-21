from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Script:

    script_idx: int | None = None
    dataset_idx: int = 0
    model_idx: int = 0
    script_code: str = ""
    sgd_best_performing_configuration: dict[str, int|float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


    def to_dict(self) -> dict :
        return {
            "script_idx": self.script_idx,
            "dataset_idx": self.dataset_idx,
            "model_idx": self.model_idx,
            "script_code": self.script_code,
            "sgd_best_performing_configuration": self.sgd_best_performing_configuration,
            "created_at": self.created_at.isoformat(),
        }
    

    @classmethod
    def from_row(cls, row: tuple):
        """Create a Script instance from a database row."""
        return cls(
            script_idx=row[0],
            dataset_idx=row[1],
            model_idx=row[2],
            script_code=row[3],
            sgd_best_performing_configuration=row[4],
            created_at=row[5]
        )