# -------------------------
# Project: Imitation Learning
# Author: benethuang
# Date: 2018.4.17
# -------------------------

import os
import json
import csv
import cv2
import time
import logging
import configparser
import random
import numpy as np
from collections import deque
from keras.models import model_from_json
from keras.preprocessing.image import array_to_img, img_to_array

model_path = "../CFM-models/Car-40800-modify-no-action-to-turn/"

ACTION_NAME_LIST = ['forward',
                    'backward',
                    'move_left',
                    'move_right',
                    'turn_left',
                    'turn_right',
                    'no_action']


class AIModel:
    def __init__(self):
        self.Init()

    def Init(self):
        # Load Model
        model_json = model_path + "test_model.json"
        model_h5 = model_path + "test_model.h5"
        with open(model_json, 'r') as jfile:
            self.model = model_from_json(json.load(jfile))

        self.logger.info("Load model: %s" %(model_h5))
        self.model.compile("adam", "categorical_crossentropy")
        self.model.load_weights(model_h5)
        self.imgList = []
        self.model.summary()
        self.w = 256
        self.h = 256

        # Counter forward
        self.forwardCount = 0
        self.lastAction   = None
        self.changeAction = 4

        # Generate Samples
        self.nSamples    = 0
        self.nMaxSamples = 20000
        self.bGenSample  = False

        if self.bGenSample:
            dataPath = "DataAugment"
            if not os.path.exists(dataPath):
                os.mkdir(dataPath)
            imgPath = dataPath + "/img"
            if not os.path.exists(imgPath):
                os.mkdir(imgPath)

            self.sampleNum = 0
            self.imgPath = imgPath
            self.dataPath = dataPath
            self.cvsPath = dataPath + "/test.csv"
            self.sampleCSVFile = open(self.cvsPath, "w")
            self.sampleCSVWriter = csv.writer(self.sampleCSVFile)
            self.sampleCSVWriter.writerow(["name", "action", "action_name"])

        return

    def GenerateSamples(self, screen, action):
        self.sampleNum = self.sampleNum + 1
        t = time.time()
        now = int(round(t*1000))
        timeStr = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(now/1000))
        savedFileName = "%s/doom-%s-%d.jpg" % (self.imgPath, timeStr, self.sampleNum)
        self.sampleCSVWriter.writerow([savedFileName, action, ACTION_NAME_LIST[action]])
        self.sampleCSVFile.flush()

        # skimage.io.imsave("hy.jpg", screen.transpose(1, 2, 0))
        # dst = ResizeImg(screen, (256, 256))
        dst = screen
        # skimage.io.imsave(savedFileName, dst)
        cv2.imwrite(savedFileName, dst)

        self.logger.info('sample: %s.' %(savedFileName))
        return

    def GetAction(self, inputImg):
        img = np.float32(inputImg)
        imgResize = cv2.resize(img, (self.w, self.h))
        imgResize = img_to_array(imgResize).transpose(0, 1, 2)
        imgResize = imgResize.astype("float32")

        imgResize = (imgResize * (1./ 255.)) - 0.5
        imgs = imgResize[None, :, :, :]
        action_id = self.model.predict(imgs, batch_size=1)
        action_list = np.argsort(-action_id, axis=1)

        # Record forward
        start = 20
        end = 45
        if action_list[0][0] == 0 and self.lastAction == 0:
            self.forwardCount += 1
        elif action_list[0][0] != 0:
            if self.forwardCount > end or self.forwardCount < start:
                self.forwardCount = 0
                self.changeAction = random.randint(4, 5)

        self.lastAction = action_list[0][0]

        if self.forwardCount >= start and self.forwardCount <= end:
            action_list[0][0] = self.changeAction
            self.forwardCount += 1
        elif self.forwardCount > 45:
            self.forwardCount = 0
            self.changeAction = random.randint(4, 5)

        self.logger.info("action: %s" %(ACTION_NAME_LIST[action_list[0][0]]))

        return action_list[0]

def Predict(folder, predictor):
    pass

if __name__ == '__main__':
    predictor = AIModel()