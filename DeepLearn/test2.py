import os

import rasterio
from matplotlib import pyplot
import numpy as np

import cv2

from PIL import Image

folder_path='earth'
image_files=[f for f in os.listdir(folder_path) if f.endswith('.png')]
for image_file in image_files:
    image_path=os.path.join(folder_path,image_file)

    with rasterio.open(image_path) as src:
        raster_data=src.read(1)
        print(src.crs)
        print(src.transform)
    pyplot.imshow(raster_data)
    pyplot.show()


    img=Image.open(image_path)
    img.show(img.resize((400,400)))

    img.close()

    img=cv2.imread(image_path)
    cv2.imshow("opencv",img)

    cv2.waitKey()
    cv2.destroyAllWindows()