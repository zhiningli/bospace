import pytest
from datetime import datetime
from src.database.object import Dataset, DatasetCode, DatasetMetaFeature

@pytest.mark.no_db
def test_dataset_class_creation():
    dataset = Dataset(dataset_idx=1, code="Hello",input_size=10, num_classes=2,  meta_features=[0.1, 0.2, 0.3], created_at=datetime.now())

    assert isinstance(dataset, Dataset)
    assert dataset.dataset_idx == 1
    assert dataset.code == "Hello"
    assert dataset.input_size == 10
    assert dataset.num_classes == 2
    assert dataset.meta_features == [0.1, 0.2, 0.3]
    assert isinstance(dataset.created_at, datetime)

@pytest.mark.no_db
def test_dataset_class_method():
    row = (1, "HI", 10, 2, [0.1, 0.2, 0.3], datetime.now())

    dataset = Dataset.from_row(row)

    assert isinstance(dataset, Dataset)
    assert dataset.dataset_idx == 1
    assert dataset.code == "HI"
    assert dataset.input_size == 10
    assert dataset.num_classes == 2
    assert dataset.meta_features == [0.1, 0.2, 0.3]
    assert isinstance(dataset.created_at, datetime)

@pytest.mark.no_db
def test_dataset_to_dict():

    dataset = Dataset(dataset_idx=1, code="Hello",input_size=10, num_classes=2, meta_features=[0.1, 0.2, 0.3], created_at=datetime.now())
    dataset_dict = dataset.to_dict()

    assert dataset_dict["dataset_idx"] == 1
    assert dataset_dict["code"] == "Hello"
    assert dataset_dict["input_size"] == 10
    assert dataset_dict["num_classes"]
    assert dataset_dict["meta_features"] == [0.1, 0.2, 0.3]
    assert isinstance(dataset_dict["created_at"], str)


@pytest.mark.no_db
def test_dataset_code_class_creation():
    dc = DatasetCode(dataset_idx=1, code="Hello")
    assert isinstance(dc, DatasetCode)
    assert dc.dataset_idx == 1
    assert dc.code == "Hello"


@pytest.mark.no_db
def test_dataset_code_class_method():
    row = [1, "HI"]

    dc = DatasetCode.from_row(row)

    assert dc is not None
    assert isinstance(dc, DatasetCode)
    assert dc.dataset_idx == 1
    assert dc.code == "HI"

@pytest.mark.no_db
def test_dataset_code_to_dict():
    dc = DatasetCode(dataset_idx=1, code="Hello")
    dict = dc.to_dict()

    assert dict is not None
    assert dict["dataset_idx"] == 1
    assert dict["code"] == "Hello"


@pytest.mark.no_db
def test_dataset_meta_featurs_class_creation():
    dm = DatasetMetaFeature(dataset_idx=1, meta_features=[0.1, 0.2, 0.3])
    assert isinstance(dm, DatasetMetaFeature)
    assert dm.dataset_idx == 1
    assert dm.meta_features == [0.1, 0.2, 0.3]

@pytest.mark.no_db
def test_dataset_metafeatures_class_method():
    row = [1, [0.1, 0.2, 0.3]]

    dm = DatasetMetaFeature.from_row(row)

    assert dm is not None
    assert isinstance(dm, DatasetMetaFeature)
    assert dm.dataset_idx == 1
    assert dm.meta_features == [0.1, 0.2, 0.3]


@pytest.mark.no_db
def test_dataset_metafeatures_to_dict():
    dm = DatasetMetaFeature(dataset_idx=1, meta_features=[0.1, 0.2, 0.3])
    dict = dm.to_dict()

    assert dict is not None
    assert dict["dataset_idx"] == 1
    assert dict["meta_features"] == [0.1, 0.2, 0.3]



