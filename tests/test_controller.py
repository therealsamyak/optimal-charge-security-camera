"""
Unit tests for intelligent controller.

Tests the ModelController class and model selection logic.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from controller.intelligent_controller import (
    ModelController,
    ControllerModelProfile,
    ControllerDecision,
)


class TestModelController:
    """Test cases for ModelController class."""

    def test_initialization(self):
        """Test controller initialization with default profiles."""
        controller = ModelController()

        # Should have all YOLOv10 models
        available_models = controller.get_available_models()
        expected_models = [
            "yolov10n",
            "yolov10s",
            "yolov10m",
            "yolov10b",
            "yolov10l",
            "yolov10x",
        ]

        for model in expected_models:
            assert model in available_models

        # Should have model profiles
        for model_name in expected_models:
            profile = controller.get_model_profile(model_name)
            assert profile is not None
            assert isinstance(profile, ControllerModelProfile)
            assert profile.name == model_name

    def test_model_profile_structure(self):
        """Test model profile data structure."""
        controller = ModelController()

        # Test yolov10n (smallest)
        profile_n = controller.get_model_profile("yolov10n")
        assert profile_n.accuracy == 39.5
        assert profile_n.latency_ms == 1.56
        assert profile_n.battery_consumption == 0.1
        assert profile_n.size_rank == 2

        # Test yolov10x (largest)
        profile_x = controller.get_model_profile("yolov10x")
        assert profile_x.accuracy == 54.4
        assert profile_x.latency_ms == 12.2
        assert profile_x.battery_consumption == 1.0
        assert profile_x.size_rank == 9

    def test_model_scoring_high_battery(self):
        """Test model scoring with high battery level."""
        controller = ModelController()

        # High battery, dirty energy - should prefer efficient models due to dirty energy
        battery_level = 90.0
        energy_cleanliness = 30.0

        scores = {}
        for model_name in controller.get_available_models():
            profile = controller.get_model_profile(model_name)
            score = controller.calculate_model_score(
                profile, battery_level, energy_cleanliness
            )
            scores[model_name] = score

        # With dirty energy, even high battery should favor efficient models
        # yolov10n (small, efficient) should score better than yolov10x (large, inefficient)
        assert (
            scores["yolov10n"] > scores["yolov10x"]
        )  # Efficient model should score better with dirty energy

    def test_model_scoring_low_battery(self):
        """Test model scoring with low battery level."""
        controller = ModelController()

        # Low battery, dirty energy - should prefer efficient models
        battery_level = 20.0
        energy_cleanliness = 30.0

        scores = {}
        for model_name in controller.get_available_models():
            profile = controller.get_model_profile(model_name)
            score = controller.calculate_model_score(
                profile, battery_level, energy_cleanliness
            )
            scores[model_name] = score

        # Smaller models should have higher scores when battery is low
        assert scores["yolov10n"] > scores["yolov10x"]
        assert scores["yolov10s"] > scores["yolov10l"]

    def test_model_scoring_clean_energy(self):
        """Test model scoring with clean energy bonus."""
        controller = ModelController()

        # Medium battery, clean energy - should get bonus for larger models
        battery_level = 60.0
        energy_cleanliness = 90.0

        scores_clean = {}
        scores_dirty = {}

        for model_name in controller.get_available_models():
            profile = controller.get_model_profile(model_name)
            scores_clean[model_name] = controller.calculate_model_score(
                profile, battery_level, energy_cleanliness
            )
            scores_dirty[model_name] = controller.calculate_model_score(
                profile, battery_level, 30.0
            )

        # Larger models should benefit more from clean energy
        clean_bonus_x = scores_clean["yolov10x"] - scores_dirty["yolov10x"]
        clean_bonus_n = scores_clean["yolov10n"] - scores_dirty["yolov10n"]

        assert clean_bonus_x > clean_bonus_n

    def test_charging_decisions(self):
        """Test charging decision logic."""
        controller = ModelController()

        # Should start charging when battery is critically low
        assert controller.should_start_charging(15.0, 50.0) is True

        # Should start charging with clean energy and medium battery
        assert controller.should_start_charging(50.0, 85.0) is True

        # Should not charge with high battery and dirty energy
        assert controller.should_start_charging(85.0, 30.0) is False

        # Should stop charging when battery is full
        assert controller.should_stop_charging(95.0, 80.0) is True

        # Should stop charging with dirty energy and reasonable battery
        assert controller.should_stop_charging(70.0, 25.0) is True

        # Should continue charging with clean energy and low battery
        assert controller.should_stop_charging(40.0, 90.0) is False

    def test_optimal_model_selection(self):
        """Test optimal model selection logic."""
        controller = ModelController()

        # Test case 1: High battery, no charging
        decision = controller.select_optimal_model(90.0, 30.0, False)

        assert isinstance(decision, ControllerDecision)
        assert decision.selected_model in controller.get_available_models()
        assert isinstance(decision.score, float)
        assert isinstance(decision.should_charge, bool)
        assert isinstance(decision.reasoning, str)
        assert decision.should_charge is False  # Should not charge with high battery

        # Test case 2: Low battery, clean energy
        decision = controller.select_optimal_model(25.0, 90.0, False)

        # Should start charging and select appropriate model
        assert decision.should_charge is True
        assert "Start charging" in decision.reasoning

        # Test case 3: Already charging
        decision = controller.select_optimal_model(40.0, 85.0, True)

        # Should continue charging
        assert decision.should_charge is True
        assert "Continue charging" in decision.reasoning

    def test_model_selection_with_requirements(self):
        """Test model selection respects user requirements."""
        controller = ModelController()

        # Test with strict requirements
        # This would require modifying the controller to accept custom requirements
        # For now, test with default requirements (80% accuracy, 100ms latency)

        decision = controller.select_optimal_model(50.0, 50.0, False)

        # Selected model should meet minimum requirements when possible

        # With medium battery and energy, should select a model that meets requirements
        # yolov10m (82% accuracy, 40ms) should be a good choice
        assert decision.selected_model in [
            "yolov10m",
            "yolov10b",
            "yolov10l",
            "yolov10x",
        ]

    def test_decision_reasoning_content(self):
        """Test that decision reasoning contains expected information."""
        controller = ModelController()

        decision = controller.select_optimal_model(60.0, 75.0, False)

        reasoning = decision.reasoning

        # Should contain model information
        assert decision.selected_model in reasoning
        assert "accuracy:" in reasoning
        assert "latency:" in reasoning

        # Should contain score
        assert f"Score: {decision.score:.2f}" in reasoning

        # Should contain battery and energy info
        assert "Battery:" in reasoning
        assert "Energy cleanliness:" in reasoning

        # Should contain charging reasoning
        assert "charging" in reasoning.lower()


class TestControllerModelProfile:
    """Test cases for ControllerModelProfile dataclass."""

    def test_model_profile_creation(self):
        """Test creating a ControllerModelProfile."""
        profile = ControllerModelProfile(
            name="test_model",
            accuracy=85.0,
            latency_ms=50.0,
            battery_consumption=0.5,
            size_rank=3,
        )

        assert profile.name == "test_model"
        assert profile.accuracy == 85.0
        assert profile.latency_ms == 50.0
        assert profile.battery_consumption == 0.5
        assert profile.size_rank == 3


class TestControllerDecision:
    """Test cases for ControllerDecision dataclass."""

    def test_controller_decision_creation(self):
        """Test creating a ControllerDecision."""
        decision = ControllerDecision(
            selected_model="yolov10m",
            should_charge=True,
            score=75.5,
            reasoning="Test reasoning",
        )

        assert decision.selected_model == "yolov10m"
        assert decision.should_charge is True
        assert decision.score == 75.5
        assert decision.reasoning == "Test reasoning"
