import pandas as pd
import os
import cv2
import shutil
import csv
import random
import time
from PIL import Image


def ResizeImg(folder):
    for (root, dirs, files) in os.walk(folder):
        for filename in files:
            file = os.path.join(root,filename)
            [shotname, extension] = os.path.splitext(file)
            print (file)

            if extension == ".jpg" or extension == ".png":
                img_png = Image.open(file)
                img_jpg = img_png.resize((1280, 720))
                img_jpg.save(file)
    return

if __name__ == '__main__':
    ResizeImg("./mark_data")
    pass