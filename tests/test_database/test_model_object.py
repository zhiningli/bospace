import pytest
from datetime import datetime
from src.database.object import Model, ModelCode, ModelFeatureVector

@pytest.mark.no_db
def test_model_class_creation():
    model = Model(model_idx=1, code="Hello", feature_vector=[0.1, 0.2, 0.3], created_at=datetime.now())

    assert isinstance(model, Model)
    assert model.model_idx == 1
    assert model.code == "Hello"
    assert model.feature_vector == [0.1, 0.2, 0.3]
    assert isinstance(model.created_at, datetime)

@pytest.mark.no_db
def test_model_class_method():
    row = (1, "HI", [0.1, 0.2, 0.3], datetime.now())

    model = Model.from_row(row)

    assert isinstance(model, Model)
    assert model.model_idx == 1
    assert model.code == "HI"
    assert model.feature_vector == [0.1, 0.2, 0.3]
    assert isinstance(model.created_at, datetime)

@pytest.mark.no_db
def test_model_to_dict():

    model = Model(model_idx=1, code="Hello", feature_vector=[0.1, 0.2, 0.3], created_at=datetime.now())
    model_dict = model.to_dict()

    assert model_dict["model_idx"] == 1
    assert model_dict["code"] == "Hello"
    assert model_dict["feature_vector"] == [0.1, 0.2, 0.3]
    assert isinstance(model_dict["created_at"], str)


@pytest.mark.no_db
def test_model_code_class_creation():
    model = ModelCode(model_idx=1, code="Hello")
    assert isinstance(model, ModelCode)
    assert model.model_idx == 1
    assert model.code == "Hello"


@pytest.mark.no_db
def test_model_code_class_method():
    row = [1, "HI"]

    model = ModelCode.from_row(row)

    assert model is not None
    assert isinstance(model, ModelCode)
    assert model.model_idx == 1
    assert model.code == "HI"

@pytest.mark.no_db
def test_model_code_to_dict():
    model = ModelCode(model_idx=1, code="Hello")
    dict = model.to_dict()

    assert dict is not None
    assert dict["model_idx"] == 1
    assert dict["code"] == "Hello"


@pytest.mark.no_db
def test_model_featurevector_class_creation():
    model = ModelFeatureVector(model_idx=1, feature_vector=[0.1, 0.2, 0.3])
    assert isinstance(model, ModelFeatureVector)
    assert model.model_idx == 1
    assert model.feature_vector == [0.1, 0.2, 0.3]

@pytest.mark.no_db
def test_model_featurevector_class_method():
    row = [1, [0.1, 0.2, 0.3]]

    model = ModelFeatureVector.from_row(row)

    assert model is not None
    assert isinstance(model, ModelFeatureVector)
    assert model.model_idx == 1
    assert model.feature_vector == [0.1, 0.2, 0.3]


@pytest.mark.no_db
def test_model_featurevector_to_dict():
    model = ModelFeatureVector(model_idx=1, feature_vector=[0.1, 0.2, 0.3])
    dict = model.to_dict()

    assert dict is not None
    assert dict["model_idx"] == 1
    assert dict["feature_vector"] == [0.1, 0.2, 0.3]




