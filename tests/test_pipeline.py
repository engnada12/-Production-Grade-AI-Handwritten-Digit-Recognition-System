"""
Automated Test Suite for Handwritten Digit Recognition Pipeline
Tests preprocessing, model loading, latency thresholds, and API formats.
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw
import base64
import io

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_handler import load_model, preprocess_image, predict_digit, get_model_path


class TestDigitRecognitionPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Loads and pre-warms the model once for all tests."""
        cls.model, cls.model_type = load_model()
        print(f"\n[Test Setup] Loaded model type: {cls.model_type}")

    def test_01_model_loading(self):
        """Verify that model loads and has valid classes."""
        self.assertIsNotNone(self.model)
        self.assertIn(self.model_type, ["CNN", "RandomForest"])
        
    def test_02_preprocessing_blank_image(self):
        """Verify preprocessing on an empty image handles gracefully without error."""
        blank = Image.new("L", (100, 100), color=255)
        flat_arr, spatial_arr, preview = preprocess_image(blank)
        
        self.assertEqual(flat_arr.shape, (1, 784))
        self.assertEqual(spatial_arr.shape, (1, 28, 28, 1))
        self.assertTrue(preview.startswith("data:image/png;base64,"))
        self.assertTrue(np.all(flat_arr >= 0.0) and np.all(flat_arr <= 1.0))

    def test_03_synthetic_drawn_digit(self):
        """Generate a synthetic drawn digit '1' and test recognition."""
        img = Image.new("L", (200, 200), color=0)
        draw = ImageDraw.Draw(img)
        # Draw vertical stroke for '1'
        draw.line([(100, 30), (100, 170)], fill=255, width=18)
        
        # Test preprocessing
        flat_arr, spatial_arr, preview = preprocess_image(img)
        self.assertEqual(flat_arr.shape, (1, 784))
        self.assertEqual(spatial_arr.shape, (1, 28, 28, 1))
        
        # Test prediction
        res = predict_digit(img, target_digits_only=True)
        self.assertEqual(res["status"], "success")
        self.assertIn(res["prediction"], range(1, 10))
        self.assertGreater(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertLess(res["timing"]["total_latency_ms"], 200.0, "Latency must be < 200ms target")

    def test_04_real_image_prediction(self):
        """Test on workspace real image file if present."""
        real_img_path = os.path.join(project_root, "..", "my_digit.png")
        if os.path.exists(real_img_path):
            res = predict_digit(real_img_path, target_digits_only=True)
            self.assertEqual(res["status"], "success")
            self.assertIn(res["prediction"], range(1, 10))
            self.assertIn("probabilities", res)
            self.assertEqual(len(res["probabilities"]), 9, "Must contain probabilities for digits 1-9")
            
            # Sum of probabilities across 1-9 should be approximately 1.0
            prob_sum = sum(res["probabilities"].values())
            self.assertAlmostEqual(prob_sum, 1.0, places=2)
            print(f"[Real Image Test] Predicted: {res['prediction']}, Confidence: {res['confidence']:.2%}, Latency: {res['timing']['total_latency_ms']}ms")

    def test_05_base64_pipeline(self):
        """Test base64 string decoding and prediction."""
        img = Image.new("RGB", (120, 120), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.line([(60, 20), (60, 100)], fill=(0, 0, 0), width=10)
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64_str = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        res = predict_digit(b64_str, target_digits_only=True)
        self.assertEqual(res["status"], "success")
        self.assertIn(res["prediction"], range(1, 10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
