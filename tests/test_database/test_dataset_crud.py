from src.database.object import Dataset, DatasetCode, DatasetMetaFeature
from src.database.crud import DatasetRepository
import logging

logger = logging.getLogger("database")
def test_create_dataset(db_transaction):
    
    meta_features = [0.1, 0.2, 0.3]
    print("meta_features", meta_features)
    dataset = DatasetRepository.create_dataset(
        code = "test_code_001",
        input_size = 100,
        num_classes=10,
        meta_features= meta_features
    )

    assert dataset is not None
    assert isinstance(dataset, Dataset)
    assert dataset.code == "test_code_001"
    assert dataset.input_size == 100
    assert dataset.num_classes == 10
    assert dataset.meta_features == [0.1, 0.2, 0.3]


def test_get_dataset_idx_with_code(db_transaction):
    dataset = DatasetRepository.create_dataset(
        code = "test_code",
        input_size = 100,
        num_classes =2,
        meta_features=[0., 9.0] 
    )

    dataset_code = DatasetRepository.get_dataset_idx_with_code(dataset.dataset_idx)
    assert isinstance(dataset_code, DatasetCode)    
    assert dataset_code is not None
    assert dataset_code.code == "test_code"
    assert dataset_code.dataset_idx == dataset.dataset_idx


def test_get_all_datasets_with_code(db_transaction):
    DatasetRepository.create_dataset("Hello", 100, 10, [0.1, 0.2])
    DatasetRepository.create_dataset("Hi", 5, 3, [0.1, 0.2, 0.3])

    datasets = DatasetRepository.get_all_datasets_with_code()
    assert datasets is not None
    assert all(isinstance(dataset, DatasetCode) for dataset in datasets)
    assert any(d.code=="Hello" for d in datasets)
    assert any(d.code == "Hi" for d in datasets)


def test_get_all_datasets_with_meta_features(db_transaction):
    DatasetRepository.create_dataset("Hello", 100, 10, [0.1, 0.2])
    DatasetRepository.create_dataset("Hi", 5, 3, [0.1, 0.2, 0.3])

    datasets = DatasetRepository.get_all_datasets_with_meta_features()
    assert datasets is not None   
    assert all(isinstance(dataset, DatasetMetaFeature) for dataset in datasets)
    assert any(d.meta_features==[0.1, 0.2] for d in datasets)
    assert any(d.meta_features==[0.1, 0.2, 0.3] for d in datasets)