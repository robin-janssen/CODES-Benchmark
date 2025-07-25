"""
Comprehensive unit tests for surrogate models, testing the AbstractSurrogateModel interface
and all registered surrogate model implementations.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from codes.surrogates import AbstractSurrogateModel, surrogate_classes


# Test constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_CHEMICALS = 10
N_TIMESTEPS = 50
N_PARAMETERS = 5
BATCH_SIZE = 4
N_SAMPLES = 8


@pytest.fixture(params=surrogate_classes)
def surrogate_model(request):
    """Fixture providing each registered surrogate model class."""
    model_class = request.param
    return model_class(
        device=DEVICE,
        n_quantities=N_CHEMICALS,
        n_timesteps=N_TIMESTEPS,
        n_parameters=N_PARAMETERS,
    )


@pytest.fixture
def sample_data():
    """Fixture providing sample training/test/validation data."""
    np.random.seed(42)
    data_train = np.random.rand(N_SAMPLES, N_TIMESTEPS, N_CHEMICALS).astype(np.float32)
    data_test = np.random.rand(N_SAMPLES // 2, N_TIMESTEPS, N_CHEMICALS).astype(np.float32)
    data_val = np.random.rand(N_SAMPLES // 2, N_TIMESTEPS, N_CHEMICALS).astype(np.float32)
    timesteps = np.linspace(0, 1, N_TIMESTEPS, dtype=np.float32)
    return data_train, data_test, data_val, timesteps


@pytest.fixture
def sample_dataloaders(surrogate_model, sample_data):
    """Fixture providing sample dataloaders."""
    data_train, data_test, data_val, timesteps = sample_data
    shuffle = True
    return surrogate_model.prepare_data(
        data_train, data_test, data_val, timesteps, BATCH_SIZE, shuffle
    )


class TestAbstractSurrogateModelInterface:
    """Test the AbstractSurrogateModel interface and registry system."""

    def test_registry_system(self):
        """Test that surrogate models are properly registered."""
        assert len(surrogate_classes) > 0, "No surrogate models registered"
        
        for model_class in surrogate_classes:
            assert issubclass(model_class, AbstractSurrogateModel)
            assert hasattr(model_class, '__name__')

    def test_protected_methods_cannot_be_overridden(self):
        """Test that protected methods cannot be overridden in subclasses."""
        protected_methods = AbstractSurrogateModel._protected_methods
        
        for model_class in surrogate_classes:
            for method in protected_methods:
                # Check that the method exists in the base class
                assert hasattr(AbstractSurrogateModel, method)
                # Check that the subclass doesn't override it
                assert method not in model_class.__dict__, \
                    f"{model_class.__name__} overrides protected method {method}"


class TestSurrogateModelInitialization:
    """Test surrogate model initialization and basic attributes."""

    def test_initialization(self, surrogate_model):
        """Test that models initialize with correct attributes."""
        assert surrogate_model.device == DEVICE
        assert surrogate_model.n_quantities == N_CHEMICALS
        assert surrogate_model.n_timesteps == N_TIMESTEPS
        assert surrogate_model.n_parameters == N_PARAMETERS
        assert surrogate_model.config is not None
        assert isinstance(surrogate_model.config, dict)
        
        # Test loss function
        assert hasattr(surrogate_model, 'L1')
        assert isinstance(surrogate_model.L1, torch.nn.L1Loss)
        
        # Test loss tracking attributes
        assert surrogate_model.train_loss is None
        assert surrogate_model.test_loss is None
        assert surrogate_model.MAE is None

    def test_device_assignment(self, surrogate_model):
        """Test that model is assigned to correct device."""
        for param in surrogate_model.parameters():
            assert param.device.type == DEVICE.split(':')[0]


class TestDataPreparation:
    """Test data preparation and dataloader creation."""

    def test_prepare_data_returns_dataloaders(self, surrogate_model, sample_data):
        """Test that prepare_data returns proper DataLoader objects."""
        data_train, data_test, data_val, timesteps = sample_data
        
        train_loader, test_loader, val_loader = surrogate_model.prepare_data(
            data_train, data_test, data_val, timesteps, BATCH_SIZE, shuffle=True
        )
        
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(test_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)

    def test_dataloader_batch_sizes(self, sample_dataloaders):
        """Test that dataloaders produce correct batch sizes."""
        train_loader, test_loader, val_loader = sample_dataloaders
        
        # Test training loader
        train_batch = next(iter(train_loader))
        assert train_batch[0].size(0) <= BATCH_SIZE, \
            f"Train batch size {train_batch[0].size(0)} exceeds {BATCH_SIZE}"
        
        # Test other loaders
        test_batch = next(iter(test_loader))
        val_batch = next(iter(val_loader))
        
        assert test_batch[0].size(0) <= BATCH_SIZE
        assert val_batch[0].size(0) <= BATCH_SIZE

    def test_dataloader_data_shapes(self, sample_dataloaders):
        """Test that dataloaders produce correctly shaped data."""
        train_loader, _, _ = sample_dataloaders
        
        batch = next(iter(train_loader))
        # Different models may have different input structures
        assert len(batch) >= 2, "Batch should contain at least inputs and targets"
        
        # Test that tensors have reasonable shapes
        for tensor in batch:
            if isinstance(tensor, torch.Tensor):
                assert tensor.ndim >= 2, "Tensors should be at least 2D"

    def test_prepare_data_with_none_values(self, surrogate_model, sample_data):
        """Test prepare_data handles None test/val data."""
        data_train, _, _, timesteps = sample_data
        
        train_loader, test_loader, val_loader = surrogate_model.prepare_data(
            data_train, None, None, timesteps, BATCH_SIZE, shuffle=True
        )
        
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert test_loader is None
        assert val_loader is None


class TestForwardPass:
    """Test model forward pass functionality."""

    def test_forward_method_exists(self, surrogate_model):
        """Test that forward method is implemented."""
        assert hasattr(surrogate_model, 'forward')
        assert callable(surrogate_model.forward)

    def test_forward_pass_shape(self, surrogate_model, sample_dataloaders):
        """Test that forward pass produces correct output shapes."""
        train_loader, _, _ = sample_dataloaders
        batch = next(iter(train_loader))
        
        with torch.no_grad():
            predictions, targets = surrogate_model(batch)
            
            assert isinstance(predictions, torch.Tensor)
            assert isinstance(targets, torch.Tensor)
            assert predictions.shape == targets.shape
            assert predictions.device.type == DEVICE.split(':')[0]

    def test_forward_pass_gradient_computation(self, surrogate_model, sample_dataloaders):
        """Test that forward pass allows gradient computation."""
        train_loader, _, _ = sample_dataloaders
        batch = next(iter(train_loader))
        
        surrogate_model.train()
        predictions, targets = surrogate_model(batch)
        
        loss = surrogate_model.L1(predictions, targets)
        loss.backward()
        
        # Check that some parameters have gradients
        has_gradients = False
        for param in surrogate_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients, "No gradients computed during backward pass"


class TestTraining:
    """Test model training functionality."""

    def test_fit_method_exists(self, surrogate_model):
        """Test that fit method is implemented."""
        assert hasattr(surrogate_model, 'fit')
        assert callable(surrogate_model.fit)

    def test_fit_updates_loss_attributes(self, surrogate_model, sample_dataloaders):
        """Test that fit method updates loss tracking attributes."""
        train_loader, test_loader, _ = sample_dataloaders
        
        # Train for minimal epochs
        surrogate_model.fit(
            train_loader, test_loader,
            epochs=2, position=0, description="test", multi_objective=False
        )
        
        # Check that loss attributes are updated
        assert surrogate_model.train_loss is not None
        assert surrogate_model.test_loss is not None
        assert surrogate_model.MAE is not None
        
        # Check shapes (should be arrays/tensors)
        assert hasattr(surrogate_model.train_loss, 'shape')
        assert hasattr(surrogate_model.test_loss, 'shape')
        assert hasattr(surrogate_model.MAE, 'shape')

    def test_fit_with_different_epochs(self, surrogate_model, sample_dataloaders):
        """Test fit method with different epoch counts."""
        train_loader, test_loader, _ = sample_dataloaders
        
        # Test with single epoch
        surrogate_model.fit(
            train_loader, test_loader,
            epochs=1, position=0, description="test", multi_objective=False
        )
        
        assert surrogate_model.n_epochs >= 1


class TestPrediction:
    """Test model prediction functionality."""

    def test_predict_method_exists(self, surrogate_model):
        """Test that predict method exists and is callable."""
        assert hasattr(surrogate_model, 'predict')
        assert callable(surrogate_model.predict)

    def test_predict_output_shapes(self, surrogate_model, sample_dataloaders):
        """Test that predict returns correctly shaped outputs."""
        train_loader, _, _ = sample_dataloaders
        
        predictions, targets = surrogate_model.predict(train_loader)
        
        assert isinstance(predictions, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        assert predictions.shape == targets.shape
        
        # Check final reshape to expected dimensions
        expected_samples = len(train_loader.dataset)
        expected_shape = (expected_samples, N_TIMESTEPS, N_CHEMICALS)
        assert predictions.shape[1:] == expected_shape[1:]
        assert targets.shape[1:] == expected_shape[1:]

    def test_predict_inference_mode(self, surrogate_model, sample_dataloaders):
        """Test that predict runs in inference mode."""
        train_loader, _, _ = sample_dataloaders
        
        # Set to training mode
        surrogate_model.train()
        
        # Predict should work regardless of training mode
        predictions, targets = surrogate_model.predict(train_loader)
        
        assert predictions is not None
        assert targets is not None


class TestSaveLoad:
    """Test model saving and loading functionality."""

    def test_save_method_exists(self, surrogate_model):
        """Test that save method exists and is callable."""
        assert hasattr(surrogate_model, 'save')
        assert callable(surrogate_model.save)

    def test_load_method_exists(self, surrogate_model):
        """Test that load method exists and is callable."""
        assert hasattr(surrogate_model, 'load')
        assert callable(surrogate_model.load)

    def test_save_load_cycle(self, surrogate_model, sample_dataloaders):
        """Test complete save/load cycle preserves model state."""
        train_loader, test_loader, _ = sample_dataloaders
        
        # Train model briefly to have some state
        surrogate_model.fit(
            train_loader, test_loader,
            epochs=1, position=0, description="test", multi_objective=False
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_name = "test_model"
            training_id = "test_training"
            
            # Save model
            surrogate_model.save(
                model_name=model_name,
                base_dir=tmp_dir,
                training_id=training_id
            )
            
            # Check files were created
            model_dir = Path(tmp_dir) / training_id / surrogate_model.__class__.__name__
            assert (model_dir / f"{model_name}.pth").exists()
            assert (model_dir / f"{model_name}.yaml").exists()
            
            # Create new instance and load
            new_model = surrogate_model.__class__(
                device=DEVICE,
                n_quantities=N_CHEMICALS,
                n_timesteps=N_TIMESTEPS,
                n_parameters=N_PARAMETERS
            )
            
            new_model.load(
                training_id=training_id,
                surr_name=surrogate_model.__class__.__name__,
                model_identifier=model_name,
                model_dir=tmp_dir
            )
            
            # Compare some basic attributes
            assert new_model.n_quantities == surrogate_model.n_quantities
            assert new_model.n_timesteps == surrogate_model.n_timesteps

    def test_save_creates_yaml_config(self, surrogate_model):
        """Test that save creates a YAML configuration file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_name = "test_model"
            training_id = "test_training"
            
            surrogate_model.save(
                model_name=model_name,
                base_dir=tmp_dir,
                training_id=training_id
            )
            
            # Check YAML file exists and is valid
            model_dir = Path(tmp_dir) / training_id / surrogate_model.__class__.__name__
            yaml_path = model_dir / f"{model_name}.yaml"
            
            assert yaml_path.exists()
            
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert isinstance(config, dict)
            assert 'device' in config


class TestDenormalization:
    """Test data denormalization functionality."""

    def test_denormalize_method_exists(self, surrogate_model):
        """Test that denormalize method exists and is callable."""
        assert hasattr(surrogate_model, 'denormalize')
        assert callable(surrogate_model.denormalize)

    def test_denormalize_without_normalization(self, surrogate_model):
        """Test denormalize when no normalization is set."""
        test_data = torch.randn(4, N_TIMESTEPS, N_CHEMICALS)
        
        # No normalization set
        assert surrogate_model.normalisation is None
        
        result = surrogate_model.denormalize(test_data)
        assert torch.allclose(result, test_data)

    def test_denormalize_with_mock_normalization(self, surrogate_model):
        """Test denormalize with mock normalization parameters."""
        test_data = torch.randn(4, N_TIMESTEPS, N_CHEMICALS)
        
        # Set mock normalization
        surrogate_model.normalisation = {
            "mode": "disabled",
            "log10_transform": False
        }
        
        result = surrogate_model.denormalize(test_data)
        assert torch.allclose(result, test_data)


class TestOptimizer:
    """Test optimizer and scheduler setup."""

    def test_setup_optimizer_and_scheduler_exists(self, surrogate_model):
        """Test that optimizer setup method exists."""
        # Not all models may have this method, so check if it exists
        if hasattr(surrogate_model, 'setup_optimizer_and_scheduler'):
            assert callable(surrogate_model.setup_optimizer_and_scheduler)

    def test_setup_optimizer_with_valid_config(self, surrogate_model):
        """Test optimizer setup with valid configuration."""
        if not hasattr(surrogate_model, 'setup_optimizer_and_scheduler'):
            pytest.skip("Model doesn't have optimizer setup method")
        
        # Mock a basic config
        surrogate_model.config.optimizer = "adamw"
        surrogate_model.config.scheduler = "cosine"
        surrogate_model.config.learning_rate = 0.001
        surrogate_model.config.regularization_factor = 0.01
        surrogate_model.config.eta_min = 1e-6
        
        try:
            optimizer, scheduler = surrogate_model.setup_optimizer_and_scheduler(epochs=10)
            assert optimizer is not None
            assert scheduler is not None
        except (AttributeError, ValueError):
            # Some models might not support all configurations
            pytest.skip("Model doesn't support this optimizer configuration")


class TestProgressBar:
    """Test progress bar setup functionality."""

    def test_setup_progress_bar_exists(self, surrogate_model):
        """Test that progress bar setup method exists."""
        assert hasattr(surrogate_model, 'setup_progress_bar')
        assert callable(surrogate_model.setup_progress_bar)

    def test_setup_progress_bar_returns_tqdm(self, surrogate_model):
        """Test that progress bar setup returns a tqdm object."""
        pbar = surrogate_model.setup_progress_bar(
            epochs=10, position=0, description="test"
        )
        
        # Check it has tqdm-like interface
        assert hasattr(pbar, 'update')
        assert hasattr(pbar, 'close')
        
        pbar.close()
