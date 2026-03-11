import unittest

from tracking.track import Track


class TrackValidationTests(unittest.TestCase):
    """
    Tests for data validation in the Track class.
    """

    def test_valid_track(self):
        """
        Test that a valid track is created correctly.
        """
        track = Track(
            id=1,
            bbox=[0.1, 0.2, 0.8, 0.9],
            class_name="person",
            confidence=0.95,
            number_of_successful_updates=5,
            frames_since_last_update=2,
            time_since_update_seconds=0.1,
        )

        self.assertEqual(track.id, 1)
        self.assertEqual(track.bbox, [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(track.class_name, "person")
        self.assertEqual(track.confidence, 0.95)
        self.assertEqual(track.number_of_successful_updates, 5)
        self.assertEqual(track.frames_since_last_update, 2)
        self.assertEqual(track.time_since_update_seconds, 0.1)

    def test_id_not_integer(self):
        """
        Test that creating a track with non-integer id raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id="1",  # type: ignore  # string instead of int
                bbox=[0.1, 0.2, 0.8, 0.9],
                class_name="person",
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track id must be an integer >= 0", str(context.exception))

    def test_id_less_than_zero(self):
        """
        Test that creating a track with id < 0 raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=-1,  # invalid id
                bbox=(0.1, 0.2, 0.8, 0.9),  # type: ignore  # tuple instead of list
                class_name="person",
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track id must be an integer >= 0", str(context.exception))

    def test_bbox_not_list(self):
        """
        Test that creating a track with bbox as non-list raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=(0.1, 0.2, 0.8, 0.9),  # type: ignore  # tuple instead of list
                class_name="person",
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track bbox must be a list of 4 floats in [0, 1]", str(context.exception))

    def test_bbox_wrong_length(self):
        """
        Test that creating a track with bbox having wrong length raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.2, 0.8],  # only 3 elements instead of 4
                class_name="person",
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track bbox must be a list of 4 floats in [0, 1]", str(context.exception))

    def test_bbox_non_float_values(self):
        """
        Test that creating a track with bbox containing non-float values raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.2, "0.8", 0.9],  # string instead of float
                class_name="person",
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track bbox must be a list of 4 floats in [0, 1]", str(context.exception))

    def test_bbox_values_out_of_range(self):
        """
        Test that creating a track with bbox values outside [0, 1] raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, -0.2, 0.8, 0.9],  # negative value
                class_name="person",
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track bbox must be a list of 4 floats in [0, 1]", str(context.exception))

    def test_bbox_x2_less_than_x1(self):
        """
        Test that creating a track with x2 < x1 raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.8, 0.2, 0.1, 0.9],  # x2 (0.1) < x1 (0.8)
                class_name="person",
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track bbox must have x2 >= x1 and y2 >= y1", str(context.exception))

    def test_bbox_y2_less_than_y1(self):
        """
        Test that creating a track with y2 < y1 raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.9, 0.8, 0.2],  # y2 (0.2) < y1 (0.9)
                class_name="person",
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track bbox must have x2 >= x1 and y2 >= y1", str(context.exception))

    def test_class_name_not_string(self):
        """
        Test that creating a track with non-string class_name raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.2, 0.8, 0.9],
                class_name=123,  # type: ignore  # integer instead of string
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track class_name must be a non-empty string", str(context.exception))

    def test_class_name_empty(self):
        """
        Test that creating a track with empty string class_name raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.2, 0.8, 0.9],
                class_name="",  # empty string
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track class_name must be a non-empty string", str(context.exception))

    def test_confidence_not_float(self):
        """
        Test that creating a track with non-float confidence raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.2, 0.8, 0.9],
                class_name="person",
                confidence="0.95",  # type: ignore  # string instead of float
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track confidence must be a float in [0, 1]", str(context.exception))

    def test_confidence_less_than_zero(self):
        """
        Test that creating a track with confidence < 0 raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.2, 0.8, 0.9],
                class_name="person",
                confidence=-0.1,  # negative confidence
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track confidence must be a float in [0, 1]", str(context.exception))

    def test_confidence_greater_than_one(self):
        """
        Test that creating a track with confidence > 1 raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.2, 0.8, 0.9],
                class_name="person",
                confidence=1.5,  # confidence greater than 1
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track confidence must be a float in [0, 1]", str(context.exception))

    def test_number_of_successful_updates_not_integer(self):
        """
        Test that creating a track with non-integer number_of_successful_updates raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.2, 0.8, 0.9],
                class_name="person",
                confidence=0.95,
                number_of_successful_updates="5",  # type: ignore  # string instead of integer
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track number_of_successful_updates must be an integer >= 0", str(context.exception))

    def test_number_of_successful_updates_negative(self):
        """
        Test that creating a track with negative number_of_successful_updates raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.2, 0.8, 0.9],
                class_name="person",
                confidence=0.95,
                number_of_successful_updates=-1,  # negative value
                frames_since_last_update=2,
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track number_of_successful_updates must be an integer >= 0", str(context.exception))

    def test_frames_since_last_update_not_integer(self):
        """
        Test that creating a track with non-integer frames_since_last_update raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.2, 0.8, 0.9],
                class_name="person",
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update="2",  # type: ignore  # string instead of integer
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track frames_since_last_update must be an integer >= 0", str(context.exception))

    def test_frames_since_last_update_negative(self):
        """
        Test that creating a track with negative frames_since_last_update raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.2, 0.8, 0.9],
                class_name="person",
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update=-1,  # negative value
                time_since_update_seconds=0.1,
            )
        self.assertIn("Track frames_since_last_update must be an integer >= 0", str(context.exception))

    def test_time_since_update_seconds_not_float(self):
        """
        Test that creating a track with non-float time_since_update_seconds raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.2, 0.8, 0.9],
                class_name="person",
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds="0.1",  # type: ignore  # string instead of float
            )
        self.assertIn("Track time_since_update_seconds must be a float >= 0", str(context.exception))

    def test_time_since_update_seconds_negative(self):
        """
        Test that creating a track with negative time_since_update_seconds raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            Track(
                id=1,
                bbox=[0.1, 0.2, 0.8, 0.9],
                class_name="person",
                confidence=0.95,
                number_of_successful_updates=5,
                frames_since_last_update=2,
                time_since_update_seconds=-0.1,  # negative value
            )
        self.assertIn("Track time_since_update_seconds must be a float >= 0", str(context.exception))
