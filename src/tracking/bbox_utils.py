"""
Bounding box utility functions that are used throughout the tracking package.
"""

from typing import List, Union


def pixels_to_normalized(  # pylint: disable=too-many-arguments  # noqa: PLR0913
    *, x1: float, y1: float, x2: float, y2: float, image_width: Union[int, float], image_height: Union[int, float]
) -> List[float]:
    """
    Convert bounding box coordinates from pixel to normalized coordinates.

    Args:
        x1: Left x coordinate in pixels
        y1: Top y coordinate in pixels
        x2: Right x coordinate in pixels
        y2: Bottom y coordinate in pixels
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels

    Returns:
        Bounding box in normalized coordinates [x1, y1, x2, y2] where values are in [0, 1]

    Raises:
        ValueError: If image dimensions are <= 0
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive")

    x1 = float(x1 / image_width)
    y1 = float(y1 / image_height)
    x2 = float(x2 / image_width)
    y2 = float(y2 / image_height)

    # verify that the values are in bounds
    if not all(0 <= c <= 1 for c in (x1, y1, x2, y2)):
        raise ValueError(
            f"After conversion, the bounding box coordinates are not in the range [0, 1]. "
            f"Input values: x1={x1}, y1={y1}, x2={x2}, y2={y2}, "
            f"image_width={image_width}, image_height={image_height}, "
            f"after conversion the output values are: x1={x1}, y1={y1}, x2={x2}, y2={y2}"
        )

    return [x1, y1, x2, y2]


def normalized_to_pixels(  # pylint: disable=too-many-arguments  # noqa: PLR0913
    *, x1: float, y1: float, x2: float, y2: float, image_width: Union[int, float], image_height: Union[int, float]
) -> List[float]:
    """
    Convert bounding box coordinates from normalized to pixel coordinates.

    Args:
        x1: Left x coordinate in normalized coordinates [0, 1]
        y1: Top y coordinate in normalized coordinates [0, 1]
        x2: Right x coordinate in normalized coordinates [0, 1]
        y2: Bottom y coordinate in normalized coordinates [0, 1]
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels

    Returns:
        Bounding box in pixel coordinates [x1, y1, x2, y2]

    Raises:
        ValueError: If image dimensions are <= 0
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive")

    x1 = float(x1 * image_width)
    y1 = float(y1 * image_height)
    x2 = float(x2 * image_width)
    y2 = float(y2 * image_height)

    # verify that the values are in bounds
    if not (all(0 <= x <= image_width for x in (x1, x2)) and all(0 <= y <= image_height for y in (y1, y2))):
        raise ValueError(
            f"After conversion, the bounding box coordinates are not in the range "
            f"[0, {image_width}] or [0, {image_height}]. Input values: x1={x1}, y1={y1}, "
            f"x2={x2}, y2={y2}, image_width={image_width}, image_height={image_height}, "
            f"after conversion the output values are: x1={x1}, y1={y1}, x2={x2}, y2={y2}"
        )

    return [x1, y1, x2, y2]
