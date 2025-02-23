from src.database.object import Model, ModelCode, ModelFeatureVector
from src.database.crud import ModelRepository

def test_create_model(db_transaction):

    model = ModelRepository.create_model(
        code = "Hello",
        feature_vector=[0.1, 0.2, 0.3]
    )

    assert model is not None
    assert isinstance(model, Model)
    assert model.code == "Hello"
    assert model.model_idx == 1
    assert model.feature_vector == [0.1, 0.2, 0.3]


def test_get_model_with_code(db_transaction):

    model = ModelRepository.create_model(
        code = "Hello",
        feature_vector=[0.1, 0.2, 0.3]
    )

    model_code = ModelRepository.get_model_with_code(model.model_idx)
    assert model_code is not None
    assert isinstance(model_code, ModelCode)
    assert model_code.model_idx == model.model_idx
    assert model_code.code == "Hello"


def test_get_all_models_with_feature_vector_only(db_transaction):

    ModelRepository.create_model(
        code = "Hello",
        feature_vector=[0.1, 0.2, 0.3]
    )  

    ModelRepository.create_model(
        code = "Hi",
        feature_vector=[0.1, 0.2]
    )  

    models = ModelRepository.get_all_models_with_feature_vector_only()
    assert models is not None
    assert all(isinstance(model, ModelFeatureVector) for model in models)
    assert any(m.feature_vector == [0.1, 0.2] for m in models)
    assert any(m.feature_vector == [0.1, 0.2, 0.3] for m in models)   


def test_get_all_models_with_code_string_only(db_transaction):

    ModelRepository.create_model(
        code = "Hello",
        feature_vector=[0.1, 0.2, 0.3]
    )  

    ModelRepository.create_model(
        code = "Hi",
        feature_vector=[0.1, 0.2]
    )  

    models = ModelRepository.get_all_models_with_code_string_only()
    assert models is not None
    assert all(isinstance(model, ModelCode) for model in models)
    assert any(m.code == "Hi" for m in models)
    assert any(m.code == "Hello" for m in models)   