from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Dataset:
    """Dataclass representing a row in the 'dataset' table"""

    dataset_idx: int | None = None
    code: str = ""
    input_size: int = ""
    num_classes: int = ""
    meta_features: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "dataset_idx": self.dataset_idx,
            "code": self.code,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "meta_features": self.meta_features,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_row(cls, row: tuple):
        return cls(
            dataset_idx = row[0],
            code = row[1],
            input_size = row[2],
            num_classes = row[3],
            meta_features = row[4],
            created_at = row[5],
        )
    
@dataclass
class DatasetMetaFeature:
    """dataclass representing parital columns, only dataset_idx, dataset_meta_feature"""
    dataset_idx: int | None = None
    meta_features: list[float] | None = None

    def to_dict(self) -> dict:
        return {
            "dataset_idx": self.dataset_idx,
            "meta_features": self.meta_features
        }
    
    @classmethod
    def from_row(cls, row:tuple):
        return cls(
            dataset_idx = row[0],
            meta_features = row[1]
        )
    
@dataclass
class DatasetCode:
    """dataclass representing parital columns, only dataset_idx, code"""
    dataset_idx: int | None = None
    code: str = ""

    def to_dict(self) -> dict:
        return {
            "dataset_idx": self.dataset_idx,
            "code": self.code
        }
    
    @classmethod
    def from_row(cls, row:tuple):
        return cls(
            dataset_idx = row[0],
            code = row[1]
        )