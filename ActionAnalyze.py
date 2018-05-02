import pandas as pd
import os
import shutil
import csv
from PIL import Image

ACTION_NAME_LIST = ['ATTACK',
                    'TURN_LEFT',
                    'TURN_LEFT+ATTACK',
                    'TURN_RIGHT',
                    'TURN_RIGHT+ATTACK',
                    'MOVE_FORWARD',
                    'MOVE_FORWARD+ATTACK',
                    'MOVE_FORWARD+TURN_LEFT',
                    'MOVE_FORWARD+TURN_LEFT+ATTACK',
                    'MOVE_FORWARD+TURN_RIGHT',
                    'MOVE_FORWARD+TURN_RIGHT+ATTACK',
                    'MOVE_BACKWARD',
                    'MOVE_BACKWARD+ATTACK',
                    'MOVE_BACKWARD+TURN_LEFT',
                    'MOVE_BACKWARD+TURN_LEFT+ATTACK',
                    'MOVE_BACKWARD+TURN_RIGHT',
                    'MOVE_BACKWARD+TURN_RIGHT+ATTACK',
                    'MOVE_LEFT',
                    'MOVE_LEFT+ATTACK',
                    'MOVE_LEFT+TURN_LEFT',
                    'MOVE_LEFT+TURN_LEFT+ATTACK',
                    'MOVE_LEFT+TURN_RIGHT',
                    'MOVE_LEFT+TURN_RIGHT+ATTACK',
                    'MOVE_RIGHT',
                    'MOVE_RIGHT+ATTACK',
                    'MOVE_RIGHT+TURN_LEFT',
                    'MOVE_RIGHT+TURN_LEFT+ATTACK',
                    'MOVE_RIGHT+TURN_RIGHT',
                    'MOVE_RIGHT+TURN_RIGHT+ATTACK']

NEW_ACTION_NAME_LIST = {
    'ATTACK':0,
    'TURN_LEFT+ATTACK':1,
    'TURN_RIGHT+ATTACK':2,
    'MOVE_RIGHT+ATTACK':3,
    'MOVE_LEFT+ATTACK':4,
    'MOVE_FORWARD+ATTACK':5,
    'MOVE_BACKWARD+ATTACK':6
}

ACTION_MAPPING = {
    'ATTACK':'ATTACK',
    'TURN_LEFT':'TURN_LEFT+ATTACK',
    'TURN_LEFT+ATTACK':'TURN_LEFT+ATTACK',
    'TURN_RIGHT':'TURN_RIGHT+ATTACK',
    'TURN_RIGHT+ATTACK':'TURN_RIGHT+ATTACK',
    'MOVE_FORWARD':'MOVE_FORWARD+ATTACK',
    'MOVE_FORWARD+ATTACK':'MOVE_FORWARD+ATTACK',
    'MOVE_FORWARD+TURN_LEFT':'MOVE_FORWARD+ATTACK',
    'MOVE_FORWARD+TURN_LEFT+ATTACK':'MOVE_FORWARD+ATTACK',
    'MOVE_FORWARD+TURN_RIGHT':'MOVE_FORWARD+ATTACK',
    'MOVE_FORWARD+TURN_RIGHT+ATTACK':'MOVE_FORWARD+ATTACK',
    'MOVE_BACKWARD':'MOVE_BACKWARD+ATTACK',
    'MOVE_BACKWARD+ATTACK':'MOVE_BACKWARD+ATTACK',
    'MOVE_BACKWARD+TURN_LEFT':'MOVE_BACKWARD+ATTACK',
    'MOVE_BACKWARD+TURN_LEFT+ATTACK':'MOVE_BACKWARD+ATTACK',
    'MOVE_BACKWARD+TURN_RIGHT':'MOVE_BACKWARD+ATTACK',
    'MOVE_BACKWARD+TURN_RIGHT+ATTACK':'MOVE_BACKWARD+ATTACK',
    'MOVE_LEFT':'MOVE_LEFT+ATTACK',
    'MOVE_LEFT+ATTACK':'MOVE_LEFT+ATTACK',
    'MOVE_LEFT+TURN_LEFT':'MOVE_LEFT+ATTACK',
    'MOVE_LEFT+TURN_LEFT+ATTACK':'MOVE_LEFT+ATTACK',
    'MOVE_LEFT+TURN_RIGHT':'MOVE_LEFT+ATTACK',
    'MOVE_LEFT+TURN_RIGHT+ATTACK':'MOVE_LEFT+ATTACK',
    'MOVE_RIGHT':'MOVE_RIGHT+ATTACK',
    'MOVE_RIGHT+ATTACK':'MOVE_RIGHT+ATTACK',
    'MOVE_RIGHT+TURN_LEFT':'MOVE_RIGHT+ATTACK',
    'MOVE_RIGHT+TURN_LEFT+ATTACK':'MOVE_RIGHT+ATTACK',
    'MOVE_RIGHT+TURN_RIGHT':'MOVE_RIGHT+ATTACK',
    'MOVE_RIGHT+TURN_RIGHT+ATTACK':'MOVE_RIGHT+ATTACK'
}

DEATHMATCH_ACTION5_NAME = [
    "ATTACK",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT"
]

def add_action_name(src):
    sampleCSVFilePath = src + "/test.csv"
    total = pd.read_csv(sampleCSVFilePath)

    sampleCSVFilePathNew = src + "/test_action.csv"
    sampleCSVFile = open(sampleCSVFilePathNew, "w")
    sampleCSVWriter = csv.writer(sampleCSVFile)
    sampleCSVWriter.writerow(["action_name", "action", "name"])
    sampleCSVFile.close()
    newTotal = pd.read_csv(sampleCSVFilePathNew)

    num_of_img = len(total)
    for index in range(num_of_img):
        img = total.iloc[index]['name'].strip()
        action = total.iloc[index]['action']
        file = os.path.basename(img)
        srcFile = src + "/img/" + file
        shotname, extension = os.path.splitext(file)
        dstFile = src + "/img/" + shotname + "_" + DEATHMATCH_ACTION5_NAME[action] + ".jpg"
        new = pd.DataFrame(
            {"name":dstFile,
             "action":action,
             "action_name":DEATHMATCH_ACTION5_NAME[action]},
             index=["0"])
        newTotal = newTotal.append(new, ignore_index=True)
        os.rename(srcFile, dstFile)

    newTotal.to_csv(sampleCSVFilePathNew, index=False, sep=',')
    return

def change_action_name(src, dst):
    total = pd.read_csv(src)
    num_of_img = len(total)

    sampleCSVFilePathNew = dst + "/test_new_action.csv"
    sampleCSVFile = open(sampleCSVFilePathNew, "w")
    sampleCSVWriter = csv.writer(sampleCSVFile)
    sampleCSVWriter.writerow(["action_name", "action", "name"])
    sampleCSVFile.close()
    newTotal = pd.read_csv(sampleCSVFilePathNew)

    for index in range(num_of_img):
        img = total.iloc[index]['name'].strip()
        action = total.iloc[index]['action']
        action_name = ACTION_NAME_LIST[action]
        file = os.path.basename(img)
        src_file = dst + "/img/" + file
        shot_name, extension = os.path.splitext(file)

        new_action_name = ACTION_MAPPING[action_name]

        dst_file = dst + "/img/" + shot_name + "_" + new_action_name + ".jpg"
        new = pd.DataFrame(
            {"name":src_file,
             "action":NEW_ACTION_NAME_LIST[new_action_name],
             "action_name":new_action_name},
             index=["0"])
        newTotal = newTotal.append(new, ignore_index=True)
        # os.rename(src_file, dst_file)
        
    newTotal.to_csv(sampleCSVFilePathNew, index=False, sep=',')    
    return

if __name__ == '__main__':
    add_action_name("../ViZDoom-Dataset/deathmatch_5action64000/data4Merge")
    # change_action_name("../ViZDoom-Dataset/validate_data100Skip2Merge4/test.csv", "../ViZDoom-Dataset/validate_data100Skip2Merge4")