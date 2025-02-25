import pytest
from unittest.mock import patch, MagicMock
from src.service.hyperparameter_evaluation_service import HPEvaluationService

def test_generate_hp_samples():
    """Test generation of Sobol hyperparameter samples."""
    service = HPEvaluationService()
    samples = service._generate_hp_samples()

    assert len(samples) == 15  # Expected 15 samples
    assert samples.shape == (15, 4)  # Each sample should have 4 parameters

@patch("src.assets.train_via_sgd.code_str", "{dataset}-{model}-{input_size}-{num_classes}")
@patch("src.database.crud.ModelRepository.get_all_models_with_code_string_only")
@patch("src.database.crud.DatasetRepository.get_dataset_idx_with_code")
@patch("src.database.crud.HPEvaluationRepository.exists_hp_evaluation")
@patch("src.middleware.component_store.ComponentStore")
def test_run_hp_evaluations_for_all_models(
    mock_component_store,
    mock_exists_evaluation,
    mock_get_dataset,
    mock_get_models
):
    """Test hyperparameter evaluation for all models with mock data."""

    # Mock dataset and model repositories
    mock_get_dataset.return_value = MagicMock(
        dataset_idx=4,
        code="dummy_dataset_code",
        input_size=32,
        num_classes=10,
        meta_features=[0.1, 0.2, 0.3]
    )

    mock_get_models.return_value = [
        MagicMock(model_idx=1, code="dummy_model_code", feature_vector=[1.0, 2.0])
    ]

    # Mock evaluation existence check
    mock_exists_evaluation.return_value = False

    # Mock ComponentStore behavior
    mock_store = MagicMock()
    mock_store.objective_func.return_value = 0.85
    mock_component_store.return_value = mock_store

    # Instantiate the service and run the evaluation
    service = HPEvaluationService()
    service.run_hp_evaluations_for_all_models()

    # Assertions to validate behavior
    assert mock_exists_evaluation.call_count == 10


@patch("src.database.crud.ModelRepository.get_all_models_with_code_string_only")
@patch("src.database.crud.DatasetRepository.get_dataset_idx_with_code")
@patch("src.database.crud.HPEvaluationRepository.exists_hp_evaluation")
@patch("src.middleware.component_store.ComponentStore")
def test_skip_existing_evaluations(
    mock_component_store,
    mock_exists_evaluation,
    mock_get_dataset,
    mock_get_models
):
    """Test that existing evaluations are skipped."""
    mock_get_dataset.return_value = MagicMock(dataset_idx=4, code="dummy_dataset_code")
    mock_get_models.return_value = [MagicMock(model_idx=1, code="dummy_model_code")]

    mock_exists_evaluation.return_value = True

    service = HPEvaluationService()
    service.run_hp_evaluations_for_all_models()

    assert mock_exists_evaluation.call_count == 10
    mock_component_store.assert_not_called()


@patch("src.database.crud.ModelRepository.get_all_models_with_code_string_only")
@patch("src.database.crud.DatasetRepository.get_dataset_idx_with_code")
@patch("src.middleware.component_store.ComponentStore")
def test_invalid_model_or_dataset(
    mock_component_store,
    mock_get_dataset,
    mock_get_models
):
    """Test behavior with invalid model or dataset."""

    mock_get_dataset.return_value = MagicMock(
        dataset_idx=4,
        code="dummy_dataset_code",
        input_size=32,
        num_classes=10
    )
    mock_get_models.return_value = []

    service = HPEvaluationService()
    service.run_hp_evaluations_for_all_models()

    mock_component_store.assert_not_called()
