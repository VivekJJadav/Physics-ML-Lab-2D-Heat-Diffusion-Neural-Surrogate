"""Tests for project setup, package importing, and configuration validation.

Verifies Milestone 1 requirements:
- Package and subpackages are importable.
- Central configuration module loads cleanly.
- Default configuration values are present, sensible, and correctly typed.
- Physical and numerical constraints are enforced via validation.
"""

import importlib
from dataclasses import FrozenInstanceError

import pytest

import physics_ml
from physics_ml.config import DEFAULT_SIMULATION_CONFIG, SimulationConfig


def test_package_import() -> None:
    """Verify that physics_ml can be imported and exports expected attributes."""
    assert hasattr(physics_ml, "__version__")
    assert isinstance(physics_ml.__version__, str)
    assert physics_ml.__version__ == "0.1.0"
    assert hasattr(physics_ml, "SimulationConfig")
    assert hasattr(physics_ml, "DEFAULT_SIMULATION_CONFIG")


@pytest.mark.parametrize(
    "subpackage_name",
    [
        "physics_ml.physics",
        "physics_ml.simulation",
        "physics_ml.data",
        "physics_ml.models",
        "physics_ml.evaluation",
    ],
)
def test_subpackages_importable(subpackage_name: str) -> None:
    """Verify that all project subpackages can be imported cleanly."""
    mod = importlib.import_module(subpackage_name)
    assert mod is not None


def test_config_module_loads() -> None:
    """Verify that the config module loads and contains the required definitions."""
    from physics_ml import config

    assert hasattr(config, "SimulationConfig")
    assert hasattr(config, "DEFAULT_SIMULATION_CONFIG")
    assert isinstance(config.DEFAULT_SIMULATION_CONFIG, config.SimulationConfig)


def test_default_configuration_values_and_sensibility() -> None:
    """Verify default simulation parameters exist and adhere to reasonable ranges."""
    cfg = DEFAULT_SIMULATION_CONFIG

    # Positive spatial dimensions
    assert cfg.plate_width == 1.0
    assert cfg.plate_height == 1.0
    assert cfg.plate_width > 0.0
    assert cfg.plate_height > 0.0

    # Grid points must be sufficient for interior differences
    assert cfg.grid_resolution_x == 51
    assert cfg.grid_resolution_y == 51
    assert cfg.grid_resolution_x >= 3
    assert cfg.grid_resolution_y >= 3

    # Diffusivity and temporal parameters must be positive
    assert cfg.thermal_diffusivity == 0.01
    assert cfg.thermal_diffusivity > 0.0

    assert cfg.simulation_duration == 1.0
    assert cfg.simulation_duration > 0.0

    assert cfg.time_step == 0.005
    assert cfg.time_step > 0.0
    assert cfg.time_step <= cfg.simulation_duration

    # Derived geometric and temporal step properties
    assert pytest.approx(cfg.dx, rel=1e-6) == 1.0 / 50.0
    assert pytest.approx(cfg.dy, rel=1e-6) == 1.0 / 50.0
    assert cfg.num_time_steps == 200
    assert cfg.fourier_number_x > 0.0
    assert cfg.fourier_number_y > 0.0


def test_configuration_types() -> None:
    """Verify that all configuration attributes have the expected primitive types."""
    cfg = DEFAULT_SIMULATION_CONFIG

    assert isinstance(cfg.plate_width, float)
    assert isinstance(cfg.plate_height, float)
    assert isinstance(cfg.grid_resolution_x, int)
    assert isinstance(cfg.grid_resolution_y, int)
    assert isinstance(cfg.thermal_diffusivity, float)
    assert isinstance(cfg.simulation_duration, float)
    assert isinstance(cfg.time_step, float)

    # Derived properties
    assert isinstance(cfg.dx, float)
    assert isinstance(cfg.dy, float)
    assert isinstance(cfg.num_time_steps, int)
    assert isinstance(cfg.fourier_number_x, float)
    assert isinstance(cfg.fourier_number_y, float)


@pytest.mark.parametrize(
    ("invalid_kwargs", "error_match"),
    [
        ({"plate_width": 0.0}, "plate_width must be positive"),
        ({"plate_width": -1.0}, "plate_width must be positive"),
        ({"plate_height": 0.0}, "plate_height must be positive"),
        ({"plate_height": -2.5}, "plate_height must be positive"),
        ({"grid_resolution_x": 2}, "grid_resolution_x must be >= 3"),
        ({"grid_resolution_x": 0}, "grid_resolution_x must be >= 3"),
        ({"grid_resolution_y": 1}, "grid_resolution_y must be >= 3"),
        ({"grid_resolution_y": -5}, "grid_resolution_y must be >= 3"),
        ({"thermal_diffusivity": 0.0}, "thermal_diffusivity must be positive"),
        ({"thermal_diffusivity": -0.01}, "thermal_diffusivity must be positive"),
        ({"simulation_duration": 0.0}, "simulation_duration must be positive"),
        ({"simulation_duration": -1.0}, "simulation_duration must be positive"),
        ({"time_step": 0.0}, "time_step must be positive"),
        ({"time_step": -0.001}, "time_step must be positive"),
        (
            {"simulation_duration": 1.0, "time_step": 2.0},
            "cannot exceed simulation_duration",
        ),
    ],
)
def test_configuration_validation_constraints(
    invalid_kwargs: dict[str, float | int], error_match: str
) -> None:
    """Verify that invalid physical or numerical parameters trigger ValueError."""
    with pytest.raises(ValueError, match=error_match):
        SimulationConfig(**invalid_kwargs)


def test_configuration_immutability() -> None:
    """Verify that SimulationConfig instances are frozen to prevent mutation."""
    cfg = SimulationConfig()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        # Attempt to modify frozen dataclass attribute
        cfg.plate_width = 2.0  # type: ignore[misc]
