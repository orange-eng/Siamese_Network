import numpy as np
from PIL import Image

from siamese import Siamese
import os
import sys
apath = os.path.abspath(os.path.dirname(sys.argv[0]))

if __name__ == "__main__":
    model = Siamese()

    image_1 = Image.open(apath+"/img/Angelic_01.png")
    image_2 = Image.open(apath+"/img/Angelic_02.png")
    probability = model.detect_image(image_1,image_2)
    print(probability)
