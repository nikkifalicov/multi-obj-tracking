from io import BufferedReader, BytesIO
from typing import Union

import numpy as np
from PIL import Image

ImageType = Union[str, bytes, Image.Image, BytesIO, BufferedReader, np.ndarray]
