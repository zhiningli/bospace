from dataclasses import dataclass

@dataclass
class DatasetEvaluationSummary:
    """Dataclass representing a row in the dataset_evaluation_summary materialized view."""
    dataset_idx: int
    elem_index: int
    avg_accuracy: float


    def to_dict(self) -> dict:
        return {
            "dataset_idx": self.dataset_idx,
            "elem_index": self.avg_accuracy,
            "avg_accuracy": self.avg_accuracy
        }
    
    @classmethod
    def from_row(cls, row):
        return cls(
            dataset_idx = row[0],
            elem_index = row[1],
            avg_accuracy = row[2]
        )

@dataclass
class ModelEvaluationSummary:
    """Dataclass representing a row in the model_evaluation_summary materialized view."""
    model_idx: int
    elem_index: int
    avg_accuracy: float

    def to_dict(self) -> dict:
        return {
            "model_idx": self.dataset_idx,
            "elem_index": self.avg_accuracy,
            "avg_accuracy": self.avg_accuracy
        }
    
    @classmethod
    def from_row(cls, row):
        return cls(
            model_idx = row[0],
            elem_index = row[1],
            avg_accuracy = row[2]
        )

