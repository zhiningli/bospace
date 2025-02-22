import pytest
from src.database.crud import DatasetRepository
from src.database.object import Dataset

def test_create_dataset(db_transaction):
    """Test dataset creation with a rtansaction"""

    dataset = DatasetRepository.create_dataset(
        code = "test_code",
        input_size= 10,
        num_classes= 4,
        meta_features= [1., 0.5]
    )

    assert dataset is not None
    assert dataset.code == "test_code"
    assert dataset.input_size == 10
    assert dataset.num_classes == 4
    assert dataset.meta_features == [1., 0.5]


def test_get_dataset(db_transaction):
    """Test handling of duplicate dataset creation"""
    created = DatasetRepository.create_dataset("fetch_test", 64, 10, [0.5, 0.6, 0.7])
    fetched = DatasetRepository.get_dataset(created.dataset_idx)
    assert fetched is not None
    assert fetched.code == created.code
    assert fetched.input_size == 64


def test_get_nonexistent_dataset(db_transaction):
    fetched = DatasetRepository.get_dataset(9999)
    assert fetched is None


def test_get_all_datasets(db_transaction):
    """Test retrieving all datasets."""
    DatasetRepository.create_dataset("dataset_1", 16, 2, [0.1, 0.1, 0.1])
    DatasetRepository.create_dataset("dataset_2", 32, 4, [0.2, 0.2, 0.2])
    datasets = DatasetRepository.get_all_dataset()
    assert len(datasets) >= 2
    assert any(d.code == "dataset_1" for d in datasets)
    assert any(d.code == "dataset_2" for d in datasets)


def test_update_meta_features(db_transaction):
    """Test updating meta_features for a dataset."""
    dataset = DatasetRepository.create_dataset("update_test", 32, 3, [0.3, 0.3, 0.3])
    success = DatasetRepository.update_meta_features(dataset.dataset_idx, [0.5, 0.5, 0.5])
    assert success is True

    updated = DatasetRepository.get_dataset(dataset.dataset_idx)
    assert updated.meta_features == [0.5, 0.5, 0.5]


def test_update_nonexistent_dataset(db_transaction):
    """Test updating a non-existent dataset."""
    success = DatasetRepository.update_meta_features(99999, [0.9, 0.9, 0.9])
    assert success is False


def test_delete_dataset(db_transaction):
    """Test dataset deletion."""
    dataset = DatasetRepository.create_dataset("delete_test", 32, 4, [0.4, 0.4, 0.4])
    success = DatasetRepository.delete_dataset(dataset.dataset_idx)
    assert success is True

    deleted = DatasetRepository.get_dataset(dataset.dataset_idx)
    assert deleted is None

def test_delete_nonexistent_dataset(db_transaction):
    """Test deletion of a non-existent dataset."""
    success = DatasetRepository.delete_dataset(99999)
    assert success is False