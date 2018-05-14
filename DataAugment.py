import pandas as pd
import os
import cv2
import shutil
import csv
import random
import time
from PIL import Image
import TrainingDefines
import ConvertToTFRecord


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


def FixActionByEnemyLocality(csv):
    total = pd.read_csv(csv)
    num_of_img = len(total)

    for index in range(num_of_img):
        action = total.iloc[index]['action']
        loc = total.iloc[index]['EnermyLoc']

        # 4: turn_left   5: turn_right
        new_action = action
        if loc < 0.49 :
            new_action = 4
        elif loc > 0.51:
            new_action = 5

        if new_action != action:
            total.loc[index, 'action'] = new_action
            total.loc[index, 'action_name'] = TrainingDefines.ACTION_NAME[new_action]

    total.to_csv(csv, index=False, sep=',')
    return


def MoveSamplesBasedCSV(csv, new_folder):
    total = pd.read_csv(csv)
    num_of_img = len(total)

    for index in range(num_of_img):
        action = total.iloc[index]['action']
        src = total.iloc[index]['name']

        [path, file] = os.path.split(src)
        [shotname, extension] = os.path.splitext(file)
        dst = new_folder + "/" + file

        shutil.move(src, dst)

        total.loc[index, 'name'] = dst

    total.to_csv(csv, index=False, sep=',')
    return

if __name__ == '__main__':
    # ResizeImg("./mark_data")
    # MoveSamplesBasedCSV("../CFM-Dataset/DataAugment-with-enemy-checking/testWithMyclassEne.csv", "../CFM-Dataset/DataAugment-with-enemy-checking/img_augment")
    # FixActionByEnemyLocality("../CFM-Dataset/DataAugment-with-enemy-checking/testWithMyclassEne.csv")

    # reading labels and file path
    train_filepaths, train_labels = ConvertToTFRecord.ReadLabelFile(
        "../CFM-Dataset/DataAugment-with-enemy-checking/testWithMyclassEne.csv")
    ConvertToTFRecord.GenerateTFRecordsWithResize(
        "../CFM-Dataset/DataAugment-with-enemy-checking/tfdata/",
        "DataAugment1",
        train_filepaths,
        train_labels,
        320,
        180)
    pass