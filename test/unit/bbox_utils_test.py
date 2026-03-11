import unittest

from tracking.bbox_utils import normalized_to_pixels, pixels_to_normalized


class BboxUtilsTests(unittest.TestCase):
    """
    Tests for bbox coordinate conversion utility functions.
    """

    def test_pixels_to_normalized_basic(self):
        """Test basic pixel to normalized conversion."""
        width, height = 100, 100

        result = pixels_to_normalized(x1=10.0, y1=20.0, x2=80.0, y2=90.0, image_width=width, image_height=height)
        expected = [0.1, 0.2, 0.8, 0.9]

        self.assertEqual(result, expected)

    def test_pixels_to_normalized_different_dimensions(self):
        """Test pixel to normalized conversion with different image dimensions."""
        width, height = 200, 400

        result = pixels_to_normalized(x1=50.0, y1=100.0, x2=150.0, y2=300.0, image_width=width, image_height=height)
        expected = [0.25, 0.25, 0.75, 0.75]

        self.assertEqual(result, expected)

    def test_pixels_to_normalized_edge_cases(self):
        """Test pixel to normalized conversion with edge cases."""
        # Full image bbox
        width, height = 100, 50

        result = pixels_to_normalized(x1=0.0, y1=0.0, x2=100.0, y2=50.0, image_width=width, image_height=height)
        expected = [0.0, 0.0, 1.0, 1.0]

        self.assertEqual(result, expected)

    def test_pixels_to_normalized_invalid_dimensions(self):
        """Test that invalid image dimensions raise ValueError."""

        with self.assertRaises(ValueError) as context:
            pixels_to_normalized(x1=10.0, y1=20.0, x2=80.0, y2=90.0, image_width=0, image_height=100)
        self.assertIn("Image width and height must be positive", str(context.exception))

        with self.assertRaises(ValueError) as context:
            pixels_to_normalized(x1=10.0, y1=20.0, x2=80.0, y2=90.0, image_width=100, image_height=-1)
        self.assertIn("Image width and height must be positive", str(context.exception))

    def test_normalized_to_pixels_basic(self):
        """Test basic normalized to pixel conversion."""
        width, height = 100, 100

        result = normalized_to_pixels(x1=0.1, y1=0.2, x2=0.8, y2=0.9, image_width=width, image_height=height)
        expected = [10.0, 20.0, 80.0, 90.0]

        self.assertEqual(result, expected)

    def test_normalized_to_pixels_different_dimensions(self):
        """Test normalized to pixel conversion with different image dimensions."""
        width, height = 200, 400

        result = normalized_to_pixels(x1=0.25, y1=0.25, x2=0.75, y2=0.75, image_width=width, image_height=height)
        expected = [50.0, 100.0, 150.0, 300.0]

        self.assertEqual(result, expected)

    def test_normalized_to_pixels_edge_cases(self):
        """Test normalized to pixel conversion with edge cases."""
        # Full image bbox
        width, height = 100, 50

        result = normalized_to_pixels(x1=0.0, y1=0.0, x2=1.0, y2=1.0, image_width=width, image_height=height)
        expected = [0.0, 0.0, 100.0, 50.0]

        self.assertEqual(result, expected)

    def test_normalized_to_pixels_invalid_dimensions(self):
        """Test that invalid image dimensions raise ValueError."""

        with self.assertRaises(ValueError) as context:
            normalized_to_pixels(x1=0.1, y1=0.2, x2=0.8, y2=0.9, image_width=0, image_height=100)
        self.assertIn("Image width and height must be positive", str(context.exception))

        with self.assertRaises(ValueError) as context:
            normalized_to_pixels(x1=0.1, y1=0.2, x2=0.8, y2=0.9, image_width=100, image_height=-1)
        self.assertIn("Image width and height must be positive", str(context.exception))

    def test_round_trip_conversion(self):
        """Test that converting pixels->normalized->pixels gives original values."""
        original_x1, original_y1, original_x2, original_y2 = 25.5, 33.7, 88.2, 91.3
        width, height = 120, 150

        # Convert to normalized and back
        normalized = pixels_to_normalized(
            x1=original_x1, y1=original_y1, x2=original_x2, y2=original_y2, image_width=width, image_height=height
        )
        back_to_pixels = normalized_to_pixels(
            x1=normalized[0],
            y1=normalized[1],
            x2=normalized[2],
            y2=normalized[3],
            image_width=width,
            image_height=height,
        )

        # Should be very close to original (allowing for floating point precision)
        self.assertAlmostEqual(original_x1, back_to_pixels[0], places=10)
        self.assertAlmostEqual(original_y1, back_to_pixels[1], places=10)
        self.assertAlmostEqual(original_x2, back_to_pixels[2], places=10)
        self.assertAlmostEqual(original_y2, back_to_pixels[3], places=10)

    def test_reverse_round_trip_conversion(self):
        """Test that converting normalized->pixels->normalized gives original values."""
        original_x1, original_y1, original_x2, original_y2 = 0.15, 0.25, 0.85, 0.95
        width, height = 200, 300

        # Convert to pixels and back
        pixels = normalized_to_pixels(
            x1=original_x1, y1=original_y1, x2=original_x2, y2=original_y2, image_width=width, image_height=height
        )
        back_to_normalized = pixels_to_normalized(
            x1=pixels[0], y1=pixels[1], x2=pixels[2], y2=pixels[3], image_width=width, image_height=height
        )

        # Should be very close to original (allowing for floating point precision)
        self.assertAlmostEqual(original_x1, back_to_normalized[0], places=10)
        self.assertAlmostEqual(original_y1, back_to_normalized[1], places=10)
        self.assertAlmostEqual(original_x2, back_to_normalized[2], places=10)
        self.assertAlmostEqual(original_y2, back_to_normalized[3], places=10)
