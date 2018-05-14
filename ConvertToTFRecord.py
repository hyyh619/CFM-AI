#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:02:26 2017

@author: yinghuang
"""

import os
import time
import pandas as pd
import tensorflow as tf
from PIL import Image
import MergeData

def encode_label(label):
    return int(label)

def ReadLabelFile(file):
    total = pd.read_csv(file)
    num_of_img = len(total)

    filepaths = []
    labels = []

    for index in range(num_of_img):
        filepath = total.iloc[index]['name'].strip()
        label = total.iloc[index]['action']

        filepaths.append(filepath)
        labels.append(encode_label(label))
        # print(action_name)
    return filepaths, labels

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def WriteTFRecords(path, prefixName, files, labels):
    i = 0
    j = 0

    for imgFile, label in zip(files, labels):
        if i == 0 :
            fileName = path + prefixName + str(j) + '.tfrecords'
            writer = tf.python_io.TFRecordWriter(fileName)

        i = i+1

        imgFile = imgFile
        img = Image.open(imgFile)
        w,h = img.size[:2]
        d = img.getbands()
        d = len(d)
        img_raw = img.tobytes()

        # if (h == 60*2 and w == 108*2 and d == 3) :
        #     example = tf.train.Example(features = tf.train.Features(feature = {
        #                                 'height': _int64_feature(h),
        #                                 'width': _int64_feature(w),
        #                                 'depth': _int64_feature(d),
        #                                 'image_raw': _bytes_feature(img_raw),
        #                                 'label':_int64_feature(label)}))
        #     writer.write(example.SerializeToString())
        # else :
        #     print(imgFile)

        example = tf.train.Example(features = tf.train.Features(feature = {
                                    'height': _int64_feature(h),
                                    'width': _int64_feature(w),
                                    'depth': _int64_feature(d),
                                    'image_raw': _bytes_feature(img_raw),
                                    'label':_int64_feature(label)}))
        writer.write(example.SerializeToString())

        if i == 10000 :
            i = 0
            writer.close()
            j = j+1

    writer.close()

def WriteTFRecordsWithResize(path, prefixName, files, labels, resizeW, resizeH):
    i = 0
    j = 0

    for imgFile, label in zip(files, labels):
        if i == 0 :
            fileName = path + prefixName + str(j) + '.tfrecords'
            writer = tf.python_io.TFRecordWriter(fileName)

        i = i+1

        imgFile = imgFile
        img = Image.open(imgFile)
        img = img.resize((resizeW, resizeH))
        w,h = img.size[:2]
        d = img.getbands()
        d = len(d)
        img_raw = img.tobytes()

        example = tf.train.Example(features = tf.train.Features(feature = {
                                    'height': _int64_feature(h),
                                    'width': _int64_feature(w),
                                    'depth': _int64_feature(d),
                                    'image_raw': _bytes_feature(img_raw),
                                    'label':_int64_feature(label)}))
        writer.write(example.SerializeToString())

        if i == 10000 :
            i = 0
            writer.close()
            j = j+1

    writer.close()

def GenerateTraingAndTestTFRecords(data_path, train_files, test_files, train_labels, test_labels):

    """
    Converts s dataset to tfrecords
    """
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    WriteTFRecords(data_path, 'test', test_files, test_labels)
    WriteTFRecords(data_path, 'train', train_files, train_labels)
    return


def GenerateTFRecordsWithResize(data_path, filePrefix, train_files, train_labels, resizeW, resizeH):
    """
    Converts s dataset to tfrecords
    """
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    WriteTFRecordsWithResize(data_path, filePrefix, train_files, train_labels, resizeW, resizeH)
    return


def PrintCurTime(strTitle):
    timeStr = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print(timeStr + '---' + strTitle)

def main(argv=None):
    PrintCurTime('begin')

    # Split samples to train and test sets.
    # MergeData.split_data_to_train_test(
    #     "../CFM-Dataset/40800-Action7/test.csv", "../CFM-Dataset/40800-Action7/train", "../CFM-Dataset/40800-Action7/test")
    # PrintCurTime('Split data')

    # MergeData.ModifyNoActionToTurn("../CFM-Dataset/40800-Action7/test/test.csv")
    # PrintCurTime('Modify no action for test')

    # MergeData.ModifyNoActionToTurn("../CFM-Dataset/40800-Action7/train/test.csv")
    # PrintCurTime('Modify no action for train')

    # Crop pictures
    # MergeData.CutSamples("../CFM-Dataset/40800-Action7/train/test.csv")
    # MergeData.CutSamples("../CFM-Dataset/40800-Action7/test/test.csv")

    # Horizontal Flip Samples
    # MergeData.HorizontalFlipSamples("../CFM-Dataset/40800-Action7/train/test_modify.csv")

    # right to left
    # MergeData.ModifyRightSamples("../CFM-Dataset/40800-Action7/train/test_modify.csv")

    # # Merge pictures
    # MergeData.generate_merge_seq_4_images(
    #     "../CFM-Dataset/40800-Action7/test/test.csv", "../CFM-Dataset/40800-Action7/validate4Merge")
    # MergeData.generate_merge_seq_4_images(
    #     "../CFM-Dataset/40800-Action7/train/test.csv", "../CFM-Dataset/40800-Action7/data4Merge")

    # Generate one hot
    MergeData.generate_one_hot_samples(
        "../CFM-Dataset/40800-Action7/test/test_modify_no_action_to_turn.csv", "../CFM-Dataset/40800-Action7/one_hot_validate_orig")
    PrintCurTime('Generate one hot samples')

    # reading labels and file path
    train_filepaths, train_labels = ReadLabelFile("../CFM-Dataset/40800-Action7/train/test_modify_no_action_to_turn.csv")
    test_filepaths, test_labels = ReadLabelFile("../CFM-Dataset/40800-Action7/test/test_modify_no_action_to_turn.csv")
    GenerateTraingAndTestTFRecords("../CFM-Dataset/40800-Action7/tfdata_orig/", train_filepaths, test_filepaths, train_labels, test_labels)
    PrintCurTime('Generate tfdata')


if __name__ == '__main__':
    main()