#!/usr/bin/env python3
"""
Unit tests for SimulationEngine class with short durations (1-5 minutes).
"""

import unittest

from src.simulation_engine import SimulationEngine, SimulationConfig, TaskGenerator
from src.controller import NaiveWeakController


class TestSimulationEngine(unittest.TestCase):
    """Unit tests for SimulationEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = SimulationConfig(
            duration_days=0,  # Will be overridden for short tests
            task_interval_seconds=5,
            time_acceleration=100,  # Fast acceleration for tests
            battery_capacity_wh=1.0,  # Small battery for quick tests
            charge_rate_watts=50.0,
        )

        # Create minimal power profiles for testing
        self.test_power_profiles = {
            "YOLOv10_N": {
                "accuracy": 39.5,
                "avg_inference_time_seconds": 0.00156,
                "model_power_mw": 602.25,
                "energy_per_inference_mwh": 1.02,
            },
            "YOLOv10_X": {
                "accuracy": 54.4,
                "avg_inference_time_seconds": 0.0122,
                "model_power_mw": 3476.75,
                "energy_per_inference_mwh": 6.06,
            },
        }

        self.controller = NaiveWeakController()

    def test_short_simulation_1_minute(self):
        """Test simulation with minimal duration."""
        # Override duration for minimal test
        self.test_config.duration_days = 0  # Minimal duration
        self.test_config.task_interval_seconds = 600  # 10-minute intervals

        engine = SimulationEngine(
            config=self.test_config,
            controller=self.controller,
            location="CA",
            season="summer",
            week=1,
            power_profiles=self.test_power_profiles,
        )

        # Run simulation
        metrics = engine.run()

        # Verify basic metrics
        self.assertGreater(metrics["total_tasks"], 0)
        self.assertGreaterEqual(metrics["completed_tasks"], 0)
        self.assertLessEqual(metrics["completed_tasks"], metrics["total_tasks"])
        self.assertGreaterEqual(metrics["task_completion_rate"], 0)
        self.assertLessEqual(metrics["task_completion_rate"], 100)

    def test_short_simulation_5_minutes(self):
        """Test simulation with minimal duration."""
        # Override duration for minimal test
        self.test_config.duration_days = 0  # Minimal duration
        self.test_config.task_interval_seconds = 300  # 5-minute intervals

        engine = SimulationEngine(
            config=self.test_config,
            controller=self.controller,
            location="CA",
            season="summer",
            week=1,
            power_profiles=self.test_power_profiles,
        )

        # Run simulation
        metrics = engine.run()

        # Should have more tasks than 1-minute simulation
        self.assertGreater(metrics["total_tasks"], 0)
        self.assertGreaterEqual(metrics["completed_tasks"], 0)

    def test_battery_depletion_scenario(self):
        """Test simulation with battery depletion."""
        # Very small battery to force depletion
        self.test_config.battery_capacity_wh = 0.001  # 1 mWh
        self.test_config.duration_days = 0  # Minimal duration
        self.test_config.task_interval_seconds = 300  # 5-minute intervals

        engine = SimulationEngine(
            config=self.test_config,
            controller=self.controller,
            location="CA",
            season="summer",
            week=1,
            power_profiles=self.test_power_profiles,
        )

        # Run simulation
        metrics = engine.run()

        # Should have missed tasks due to battery depletion
        self.assertGreater(metrics["missed_deadlines"], 0)
        self.assertLess(metrics["task_completion_rate"], 100)

    def test_different_task_intervals(self):
        """Test simulation with different task intervals."""
        intervals = [1, 5, 10]

        for interval in intervals:
            with self.subTest(interval=interval):
                self.test_config.task_interval_seconds = interval
                self.test_config.duration_days = 0  # Minimal duration

                engine = SimulationEngine(
                    config=self.test_config,
                    controller=self.controller,
                    location="CA",
                    season="summer",
                    week=1,
                    power_profiles=self.test_power_profiles,
                )

                metrics = engine.run()

                # Should complete successfully
                self.assertGreater(metrics["total_tasks"], 0)
                self.assertGreaterEqual(metrics["task_completion_rate"], 0)

    def test_time_acceleration(self):
        """Test time acceleration functionality."""
        accelerations = [1, 10, 100]

        for accel in accelerations:
            with self.subTest(acceleration=accel):
                self.test_config.time_acceleration = accel
                self.test_config.duration_days = 0  # Minimal duration
                self.test_config.task_interval_seconds = 600  # 10-minute intervals

                engine = SimulationEngine(
                    config=self.test_config,
                    controller=self.controller,
                    location="CA",
                    season="summer",
                    week=1,
                    power_profiles=self.test_power_profiles,
                )

                import time

                start_time = time.time()
                metrics = engine.run()
                elapsed_time = time.time() - start_time

                # Should complete (acceleration affects runtime)
                self.assertGreater(metrics["total_tasks"], 0)
                # Higher acceleration should result in faster completion
                self.assertLess(elapsed_time, 10)  # Should complete within 10 seconds


class TestTaskGenerator(unittest.TestCase):
    """Unit tests for TaskGenerator class."""

    def test_deterministic_seed(self):
        """Test TaskGenerator with deterministic seed for reproducible results."""
        seed = 42
        generator1 = TaskGenerator(seed=seed)
        generator2 = TaskGenerator(seed=seed)

        # Generate tasks with same seed
        tasks1 = []
        tasks2 = []

        for i in range(10):
            task1 = generator1.generate_task(float(i), 80.0, 2000.0)
            task2 = generator2.generate_task(float(i), 80.0, 2000.0)
            tasks1.append(task1)
            tasks2.append(task2)

        # Should generate identical sequences
        for i, (t1, t2) in enumerate(zip(tasks1, tasks2)):
            if t1 is not None and t2 is not None:
                self.assertEqual(t1.accuracy_requirement, t2.accuracy_requirement)
                self.assertEqual(t1.latency_requirement, t2.latency_requirement)
            else:
                # Both should be None or both should be tasks
                self.assertEqual(t1 is None, t2 is None)

    def test_task_generation_ranges(self):
        """Test that generated tasks are within expected ranges."""
        generator = TaskGenerator(seed=123)

        accuracy_requirements = []
        latency_requirements = []

        # Generate many tasks to test ranges
        for i in range(100):
            task = generator.generate_task(float(i), 80.0, 2000.0)
            if task is not None:
                accuracy_requirements.append(task.accuracy_requirement)
                latency_requirements.append(task.latency_requirement)

        # Check ranges
        if accuracy_requirements:
            self.assertGreaterEqual(min(accuracy_requirements), 70.0)
            self.assertLessEqual(max(accuracy_requirements), 95.0)

        if latency_requirements:
            self.assertGreaterEqual(min(latency_requirements), 1000.0)
            self.assertLessEqual(max(latency_requirements), 3000.0)

    def test_task_probability(self):
        """Test that approximately 90% of calls generate tasks."""
        generator = TaskGenerator(seed=456)

        task_count = 0
        total_calls = 1000

        for i in range(total_calls):
            task = generator.generate_task(float(i), 80.0, 2000.0)
            if task is not None:
                task_count += 1

        # Should be approximately 90% (allowing some variance)
        task_percentage = (task_count / total_calls) * 100
        self.assertGreater(task_percentage, 85)
        self.assertLess(task_percentage, 95)


if __name__ == "__main__":
    unittest.main()
