#!/usr/bin/env python3
"""
Main entry point for Optimal Charge Security Camera system.

Real-time webcam processing with intelligent model selection and battery management.
"""

import sys
import time
import signal
import argparse
import atexit
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from controller.hybrid_controller import HybridController, ControllerType
from models.yolo_manager import YOLOv10Manager
from sensors.mock_battery import MockBatterySensor
from sensors.mock_energy import MockEnergySensor
from utils.helpers import setup_logging


class SecurityCameraSystem:
    """Main security camera system with real-time processing."""

    def __init__(self, args):
        """Initialize the camera system."""
        self.args = args
        self.running = False
        self.frame_count = 0

        # Setup logging
        setup_logging(args.verbose)

        # Initialize components
        if args.controller == "rule_based":
            from controller.intelligent_controller import ModelController

            self.controller = ModelController()
        else:
            self.controller = HybridController(ControllerType(args.controller))
        self.model_manager = YOLOv10Manager()

        # Initialize sensors
        if args.mock_battery is not None:
            self.battery_sensor = MockBatterySensor(initial_battery=args.mock_battery)
        else:
            self.battery_sensor = MockBatterySensor()

        # Register cleanup function for atexit
        atexit.register(self.cleanup)

        self.energy_sensor = MockEnergySensor()

        # Initialize webcam
        self.cap = None
        self._init_webcam()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Security camera system initialized")

    def _init_webcam(self):
        """Initialize webcam capture."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open webcam")

            # Set webcam properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            logger.info("Webcam initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize webcam: {e}")
            raise

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _get_frame(self) -> Optional[np.ndarray]:
        """Get frame from webcam."""
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from webcam")
            return None

        return frame

    def _process_frame(self, frame: np.ndarray) -> dict:
        """Process a single frame through the pipeline."""
        # Get current sensor readings
        battery_level = self.battery_sensor.get_battery_percentage()
        energy_cleanliness = self.energy_sensor.get_energy_cleanliness()
        is_charging = self.battery_sensor.is_charging()

        # Make controller decision
        if hasattr(self.controller, "select_optimal_model"):
            decision = self.controller.select_optimal_model(
                battery_level=battery_level,
                energy_cleanliness=energy_cleanliness,
                is_charging=is_charging,
            )
        else:
            # Hybrid controller interface
            decision = self.controller.select_optimal_model(
                battery_level=battery_level,
                energy_cleanliness=energy_cleanliness,
                is_charging=is_charging,
                user_requirements={
                    "min_accuracy": self.args.min_accuracy,
                    "max_latency_ms": self.args.max_latency,
                    "run_frequency_ms": self.args.interval_ms,
                },
            )

        # Handle charging control
        if self.args.enable_charging:
            if decision.should_charge and not is_charging:
                self.battery_sensor.start_charging()
                logger.info("Started charging (clean energy available)")
            elif not decision.should_charge and is_charging:
                self.battery_sensor.stop_charging()
                logger.info("Stopped charging")

        # Run inference with selected model
        results, metrics = self.model_manager.run_inference(
            frame, decision.selected_model
        )

        # Update battery consumption
        if metrics.get("battery_consumption", 0) > 0:
            self.battery_sensor.consume_battery(metrics["battery_consumption"])

        # Update controller with performance outcomes
        if hasattr(self.controller, "update_performance_outcome"):
            self.controller.update_performance_outcome(
                accuracy=metrics.get("confidence", 0) * 100,
                latency_ms=metrics.get("latency_ms", 0),
                battery_consumption=metrics.get("battery_consumption", 0),
            )

        # Prepare processing results
        processing_results = {
            "frame": frame,
            "decision": decision,
            "metrics": metrics,
            "battery_level": battery_level,
            "energy_cleanliness": energy_cleanliness,
            "is_charging": is_charging,
            "detection_count": 0,
            "detection_label": "no detection",
        }

        # Extract detection information from YOLO results
        # Use the metrics already extracted by the YOLO manager
        if metrics.get("has_detection", False) and metrics.get("label") == "person":
            # Human detected
            processing_results["detection_count"] = metrics.get("detection_count", 1)
            processing_results["detection_label"] = "person detected"
        elif metrics.get("has_detection", False):
            # Other objects detected but not human
            processing_results["detection_count"] = 0
            processing_results["detection_label"] = "no human detected"
        else:
            # No objects detected
            processing_results["detection_count"] = 0
            processing_results["detection_label"] = "no detection"

        return processing_results

    def _display_frame(self, processing_results: dict):
        """Display frame with overlay information."""
        frame = processing_results["frame"].copy()
        decision = processing_results["decision"]
        metrics = processing_results["metrics"]

        # Prepare overlay text
        overlay_lines = [
            f"Model: {decision.selected_model}",
            f"Score: {decision.score:.1f}",
            f"Battery: {processing_results['battery_level']:.1f}% {'⚡' if processing_results['is_charging'] else ''}",
            f"Energy: {processing_results['energy_cleanliness']:.1f}% clean",
            f"Detection: {processing_results['detection_label']} ({processing_results['detection_count']})",
            f"Latency: {metrics.get('latency_ms', 0):.1f}ms",
            f"Confidence: {metrics.get('confidence', 0):.2f}",
        ]

        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Add text
        y_offset = 30
        for line in overlay_lines:
            cv2.putText(
                frame,
                line,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            y_offset += 25

        # Add charging indicator
        if processing_results["is_charging"]:
            cv2.putText(
                frame,
                "CHARGING",
                (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # Display frame
        cv2.imshow("Optimal Charge Security Camera", frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            self.running = False
        elif key == ord("c"):  # 'c' to toggle charging
            if self.args.enable_charging:
                if processing_results["is_charging"]:
                    self.battery_sensor.stop_charging()
                else:
                    self.battery_sensor.start_charging()

    def run(self):
        """Main processing loop."""
        logger.info("Starting real-time security camera processing...")
        self.running = True

        try:
            while self.running:
                start_time = time.time()

                # Get frame
                frame = self._get_frame()
                if frame is None:
                    logger.warning("Failed to get frame, retrying...")
                    time.sleep(0.1)
                    continue

                # Process frame
                processing_results = self._process_frame(frame)

                # Display results
                if not self.args.no_display:
                    self._display_frame(processing_results)

                # Log periodically
                self.frame_count += 1
                if self.frame_count % 30 == 0:  # Log every 30 frames
                    logger.info(
                        f"Frame {self.frame_count}: {processing_results['decision'].selected_model} | "
                        f"Battery: {processing_results['battery_level']:.1f}% | "
                        f"Energy: {processing_results['energy_cleanliness']:.1f}% | "
                        f"Detection: {processing_results['detection_label']}"
                    )

                # Maintain frame rate
                processing_time = (time.time() - start_time) * 1000
                sleep_time = max(0, (self.args.interval_ms - processing_time) / 1000)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up resources...")

        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # Save training data if collected
        try:
            if hasattr(self.controller, "save_training_data"):
                self.controller.save_training_data()
        except Exception:
            pass

        logger.info("Security camera system stopped")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimal Charge Security Camera - Real-time Webcam Processing"
    )

    # Source options
    parser.add_argument(
        "--source",
        default="webcam",
        choices=["webcam"],
        help="Input source type (default: webcam)",
    )

    # Runtime options
    parser.add_argument(
        "-i",
        "--interval-ms",
        type=int,
        default=2000,
        help="Processing interval in milliseconds (default: 2000)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument("--no-display", action="store_true", help="Disable GUI display")

    # Controller options
    parser.add_argument(
        "-c",
        "--controller",
        default="rule_based",
        choices=["rule_based", "ml_based", "hybrid"],
        help="Controller type (default: rule_based)",
    )
    parser.add_argument(
        "--enable-charging",
        action="store_true",
        default=True,
        help="Enable battery charging control",
    )
    parser.add_argument(
        "--disable-charging",
        dest="enable_charging",
        action="store_false",
        help="Disable battery charging control",
    )

    # Performance requirements
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=80.0,
        help="Minimum accuracy requirement (default: 80.0)",
    )
    parser.add_argument(
        "--max-latency",
        type=float,
        default=100.0,
        help="Maximum latency requirement in ms (default: 100.0)",
    )

    # Sensor options
    parser.add_argument(
        "--mock-battery", type=float, help="Initial mock battery level (0-100)"
    )
    parser.add_argument(
        "--min-battery",
        type=float,
        default=20.0,
        help="Minimum battery threshold (0-100) (default: 20.0)",
    )
    parser.add_argument(
        "--max-battery",
        type=float,
        default=90.0,
        help="Maximum battery threshold (0-100) (default: 90.0)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Validate arguments
    if args.min_accuracy < 0 or args.min_accuracy > 100:
        logger.error("Minimum accuracy must be between 0 and 100")
        sys.exit(1)

    if args.max_latency <= 0:
        logger.error("Maximum latency must be positive")
        sys.exit(1)

    if args.interval_ms <= 0:
        logger.error("Interval must be positive")
        sys.exit(1)

    if args.mock_battery is not None and (
        args.mock_battery < 0 or args.mock_battery > 100
    ):
        logger.error("Mock battery level must be between 0 and 100")
        sys.exit(1)

    try:
        # Create and run system
        system = SecurityCameraSystem(args)
        system.run()

    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
