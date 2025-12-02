#!/usr/bin/env python3
"""
Unit tests for Wh-based battery operations and charging behavior.
"""

import unittest

from src.battery import Battery


class TestBatteryIntegration(unittest.TestCase):
    """Test Wh-based battery operations with power profile conversions."""

    def setUp(self):
        """Set up test fixtures."""
        self.battery = Battery(capacity_wh=5.0, charge_rate_watts=100.0)

    def test_initialization(self):
        """Test battery initialization with Wh units."""
        self.assertEqual(self.battery.capacity_wh, 5.0)
        self.assertEqual(self.battery.charge_rate_watts, 100.0)
        self.assertEqual(self.battery.get_level_wh(), 5.0)
        self.assertEqual(self.battery.get_percentage(), 100.0)

    def test_power_conversion_mw_to_w(self):
        """Test conversion from mW to W for power profiles."""
        # Test with typical YOLOv10 power consumption
        power_mw = 602.25  # YOLOv10_N model power in mW
        expected_power_w = 0.60225  # Should be 602.25 / 1000

        # Simulate discharge
        duration_seconds = 1.0
        initial_energy = self.battery.get_level_wh()

        success = self.battery.discharge(
            power_mw=power_mw,
            duration_seconds=duration_seconds,
            clean_energy_percentage=0.0,
        )

        self.assertTrue(success)

        # Check energy consumption: power_w * duration_seconds / 3600
        expected_energy_wh = expected_power_w * (duration_seconds / 3600)
        final_energy = self.battery.get_level_wh()
        actual_energy_used = initial_energy - final_energy

        self.assertAlmostEqual(actual_energy_used, expected_energy_wh, places=6)

    def test_energy_calculation_accuracy(self):
        """Test energy calculation accuracy for different power levels."""
        test_cases = [
            {
                "power_mw": 1000,
                "duration": 3600,
                "expected_wh": 1.0,
            },  # 1W for 1 hour = 1Wh
            {
                "power_mw": 2000,
                "duration": 1800,
                "expected_wh": 1.0,
            },  # 2W for 30 min = 1Wh
            {
                "power_mw": 500,
                "duration": 7200,
                "expected_wh": 1.0,
            },  # 0.5W for 2 hours = 1Wh
            {
                "power_mw": 3476.75,
                "duration": 1,
                "expected_wh": 0.000965764,
            },  # YOLOv10_X for 1 second
        ]

        for case in test_cases:
            with self.subTest(case=case):
                # Reset battery
                self.battery = Battery(capacity_wh=5.0, charge_rate_watts=100.0)

                initial_energy = self.battery.get_level_wh()

                success = self.battery.discharge(
                    power_mw=case["power_mw"],
                    duration_seconds=case["duration"],
                    clean_energy_percentage=0.0,
                )

                self.assertTrue(success)

                final_energy = self.battery.get_level_wh()
                actual_energy_used = initial_energy - final_energy

                self.assertAlmostEqual(
                    actual_energy_used,
                    case["expected_wh"],
                    places=6,
                    msg=f"Failed for power_mw={case['power_mw']}, duration={case['duration']}",
                )

    def test_clean_energy_tracking(self):
        """Test clean energy percentage tracking."""
        # Test with 50% clean energy
        power_mw = 1000  # 1W
        duration_seconds = 3600  # 1 hour

        initial_total = self.battery.get_total_energy_used_wh()
        initial_clean = self.battery.get_total_clean_energy_used_wh()

        success = self.battery.discharge(
            power_mw=power_mw,
            duration_seconds=duration_seconds,
            clean_energy_percentage=50.0,
        )

        self.assertTrue(success)

        final_total = self.battery.get_total_energy_used_wh()
        final_clean = self.battery.get_total_clean_energy_used_wh()

        total_used = final_total - initial_total
        clean_used = final_clean - initial_clean

        # Should be 1Wh total, 0.5Wh clean
        self.assertAlmostEqual(total_used, 1.0, places=6)
        self.assertAlmostEqual(clean_used, 0.5, places=6)

    def test_charging_behavior(self):
        """Test charging behavior between controller decisions."""
        # Discharge some energy first
        self.battery.discharge(
            power_mw=1000,  # 1W
            duration_seconds=1800,  # 30 minutes
            clean_energy_percentage=0.0,
        )

        # Should have used 0.5Wh
        self.assertAlmostEqual(self.battery.get_level_wh(), 4.5, places=6)

        # Test charging for 30 minutes (task interval)
        charge_duration = 1800  # 30 minutes
        initial_level = self.battery.get_level_wh()

        energy_added = self.battery.charge(charge_duration)

        # Expected: 100W * 1800s / 3600s/h = 50Wh
        # But battery can only accept up to capacity (5Wh total)
        # Available space: 5.0 - 4.5 = 0.5Wh
        expected_added = 5.0 - initial_level  # Available space
        self.assertAlmostEqual(energy_added, expected_added, places=8)

        # Should be fully charged
        final_level = self.battery.get_level_wh()
        self.assertEqual(final_level, 5.0)

    def test_charging_when_full(self):
        """Test charging when battery is already full."""
        # Battery starts full
        self.assertEqual(self.battery.get_percentage(), 100.0)

        energy_added = self.battery.charge(3600)  # Charge for 1 hour

        # Should add no energy
        self.assertEqual(energy_added, 0.0)
        self.assertEqual(self.battery.get_level_wh(), 5.0)
        self.assertEqual(self.battery.get_percentage(), 100.0)

    def test_insufficient_battery(self):
        """Test discharge when insufficient battery."""
        # Try to use more energy than available
        power_mw = 5000  # 5W
        duration_seconds = 3600 * 2  # 2 hours = 10Wh needed

        initial_level = self.battery.get_level_wh()

        success = self.battery.discharge(
            power_mw=power_mw,
            duration_seconds=duration_seconds,
            clean_energy_percentage=0.0,
        )

        # Should fail
        self.assertFalse(success)

        # Battery level should be unchanged
        final_level = self.battery.get_level_wh()
        self.assertEqual(final_level, initial_level)

    def test_battery_depletion_scenario(self):
        """Test complete battery depletion and recovery."""
        # Discharge until empty
        power_mw = 1000  # 1W
        total_duration = 18000  # 5 hours = 5Wh needed

        # Discharge in steps
        step_duration = 3600  # 1 hour steps
        steps_completed = 0

        for i in range(0, total_duration, step_duration):
            success = self.battery.discharge(
                power_mw=power_mw,
                duration_seconds=step_duration,
                clean_energy_percentage=0.0,
            )

            if success:
                steps_completed += 1
            else:
                break

        # Should complete 5 steps (5 hours) then fail on 6th
        self.assertEqual(steps_completed, 5)
        self.assertAlmostEqual(self.battery.get_level_wh(), 0.0, places=6)

        # Try one more discharge - should fail
        success = self.battery.discharge(
            power_mw=power_mw,
            duration_seconds=step_duration,
            clean_energy_percentage=0.0,
        )
        self.assertFalse(success)

        # Test recovery by charging
        charge_energy = self.battery.charge(1800)  # Charge for 30 minutes
        # But limited by battery capacity (5Wh available)
        expected_limited = 5.0  # Battery capacity
        self.assertAlmostEqual(charge_energy, expected_limited, places=8)

        # But limited by battery capacity
        self.assertEqual(self.battery.get_level_wh(), 5.0)

    def test_energy_tracking_reset(self):
        """Test energy usage tracking reset."""
        # Use some energy
        self.battery.discharge(
            power_mw=1000, duration_seconds=3600, clean_energy_percentage=50.0
        )

        # Verify energy was tracked
        self.assertGreater(self.battery.get_total_energy_used_wh(), 0)
        self.assertGreater(self.battery.get_total_clean_energy_used_wh(), 0)

        # Reset tracking
        self.battery.reset_energy_tracking()

        # Should be zero
        self.assertEqual(self.battery.get_total_energy_used_wh(), 0.0)
        self.assertEqual(self.battery.get_total_clean_energy_used_wh(), 0.0)

    def test_power_profile_integration(self):
        """Test integration with actual power profile data."""
        # Load actual power profiles
        import json

        with open("results/power_profiles.json", "r") as f:
            power_profiles = json.load(f)

        # Test with each model
        for model_name, profile in power_profiles.items():
            with self.subTest(model=model_name):
                # Reset battery
                self.battery = Battery(capacity_wh=5.0, charge_rate_watts=100.0)

                power_mw = profile["model_power_mw"]
                duration_seconds = profile["avg_inference_time_seconds"]

                initial_energy = self.battery.get_level_wh()

                success = self.battery.discharge(
                    power_mw=power_mw,
                    duration_seconds=duration_seconds,
                    clean_energy_percentage=25.0,
                )

                self.assertTrue(success)

                # Check energy matches profile calculation
                # Energy = power_w * duration_seconds / 3600
                power_w = profile["model_power_mw"] / 1000  # Convert mW to W
                expected_energy_wh = (
                    power_w * profile["avg_inference_time_seconds"] / 3600
                )
                final_energy = self.battery.get_level_wh()
                actual_energy_used = initial_energy - final_energy

                self.assertAlmostEqual(
                    actual_energy_used,
                    expected_energy_wh,
                    places=8,
                    msg=f"Energy mismatch for {model_name}: expected {expected_energy_wh}, got {actual_energy_used}",
                )

    def test_charging_between_decisions(self):
        """Test charging behavior between controller decisions (task intervals)."""
        # Simulate task interval scenario
        task_interval = 5  # 5 seconds between decisions

        # Start with partial charge
        self.battery = Battery(capacity_wh=5.0, charge_rate_watts=100.0)
        self.battery.discharge(
            power_mw=1000,  # 1W
            duration_seconds=14400,  # 4 hours = 4Wh used
            clean_energy_percentage=0.0,
        )

        self.assertAlmostEqual(self.battery.get_level_wh(), 1.0, places=6)

        # Controller decides to charge - charge for one task interval
        charge_energy = self.battery.charge(task_interval)
        expected_charge = 100.0 * (task_interval / 3600)  # 100W * 5s / 3600s/h

        self.assertAlmostEqual(charge_energy, expected_charge, places=6)

        # Should have added small amount of charge
        final_level = self.battery.get_level_wh()
        expected_final = 1.0 + expected_charge
        self.assertAlmostEqual(final_level, expected_final, places=6)


if __name__ == "__main__":
    unittest.main()
