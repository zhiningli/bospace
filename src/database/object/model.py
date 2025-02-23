from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Model:
    """Dataclass representing a row in the 'models' table """
    model_idx: int | None = None
    code: str = ""
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
    
@dataclass
class ModelFeatureVector:
    """dataclass representing parital columns, only model_idx, feature_vector"""
    model_idx: int | None = None
    feature_vector: list[float] | None = None

    def to_dict(self) -> dict:
        return {
            "model_idx": self.model_idx,
            "feature_vector": self.feature_vector
        }
    
    @classmethod
    def from_row(cls, row:tuple):
        return cls(
            model_idx = row[0],
            feature_vector = row[1]
        )
    
@dataclass
class ModelCode:
    """dataclass representing parital columns, only model_idx, code"""
    model_idx: int | None = None
    code: str = ""

    def to_dict(self) -> dict:
        return {
            "model_idx": self.model_idx,
            "code": self.code
        }
    
    @classmethod
    def from_row(cls, row:tuple):
        return cls(
            model_idx = row[0],
            code = row[1]
        )