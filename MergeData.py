import pandas as pd
import os
import shutil
import csv
import random
import time
from PIL import Image

dataNum = 53

DEATHMATCH_ACTION5_NAME = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "NO_ACTION"
]


def generate_samples(start, end, src, dst, skip):
    if not os.path.exists(dst):
        os.mkdir(dst)
    imgDst = dst + "/img"    
    if not os.path.exists(imgDst):
        os.mkdir(imgDst)

    sampleCSVFilePath = dst + "/test.csv"
    sampleCSVFile = open(sampleCSVFilePath, "w")
    sampleCSVWriter = csv.writer(sampleCSVFile)
    sampleCSVWriter.writerow(["action", "name"])
    sampleCSVFile.close()

    total = pd.read_csv(sampleCSVFilePath)

    for i in range(start, end):
        folder = "data%d" % (i)
        csvFilePath = src + "/" + folder + "/test.csv"
        data = pd.read_csv(csvFilePath)
        # total = pd.concat([data, total])

        num_of_img = len(data)
        for index in range(num_of_img):
            if (index % skip) == 0 :
                img = data.iloc[index]['name'].strip()
                file = os.path.basename(img)
                srcfile = src + "/" + folder + "/img/" + file
                dstfile = imgDst + "/" + file
                # shutil.move(srcfile, dstfile)
                shutil.copy(srcfile, dstfile)
                new = pd.DataFrame({"name":dstfile, "action":data.iloc[index]['action']},index=["0"])
                total = total.append(new, ignore_index=True)

    total.to_csv(sampleCSVFilePath, index=False, sep=',')
    return

def generate_samples_4_merge(start, end, src, dst, skip):
    if not os.path.exists(dst):
        os.mkdir(dst)
    imgDst = dst + "/img"    
    if not os.path.exists(imgDst):
        os.mkdir(imgDst)

    sampleCSVFilePath = dst + "/test.csv"
    sampleCSVFile = open(sampleCSVFilePath, "w")
    sampleCSVWriter = csv.writer(sampleCSVFile)
    sampleCSVWriter.writerow(["action", "name"])
    sampleCSVFile.close()

    total = pd.read_csv(sampleCSVFilePath)

    for i in range(start, end):
        folder = "data%d" % (i)
        csvFilePath = src + "/" + folder + "/test.csv"
        data = pd.read_csv(csvFilePath)
        # total = pd.concat([data, total])[]

        imgList = []

        num_of_img = len(data)
        for index in range(num_of_img):
            if (index % skip) == 0 :
                img = data.iloc[index]['name'].strip()
                file = os.path.basename(img)
                srcfile = src + "/" + folder + "/img/" + file
                img = Image.open(srcfile)
                imgList.append(img)

                if len(imgList) < 4:
                    continue

                action = data.iloc[index]['action']
                w, h = img.size[0], img.size[1]
                mergeimg = Image.new('RGB', (w * 2, h*2), 0xffffff)    

                mergeimg.paste(imgList[0], (0, 0))
                mergeimg.paste(imgList[1], (w, 0))
                mergeimg.paste(imgList[2], (0, h))
                mergeimg.paste(imgList[3], (w, h))

                dstfile = "%s/%d_%d_%d.jpg" % (imgDst, i, index, action)
                mergeimg.save(dstfile)
                # shutil.move(srcfile, dstfile)
                # shutil.copy(srcfile, dstfile)
                new = pd.DataFrame({"name":dstfile, "action":data.iloc[index]['action']},index=["0"])
                total = total.append(new, ignore_index=True)
                imgList.pop(0)

    total.to_csv(sampleCSVFilePath, index=False, sep=',')
    return

def generate_one_hot_samples(csv, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)

    data = pd.read_csv(csv)
    num_of_img = len(data)
    for index in range(num_of_img):
        img = data.iloc[index]['name'].strip()
        action = data.iloc[index]['action']
        actionPath = "%s/%d" % (dst, action)
        if not os.path.exists(actionPath):
            os.mkdir(actionPath)
        file = os.path.basename(img)
        srcfile = "./" + img
        dstfile = actionPath + "/" + file
        # shutil.move(srcfile, dstfile)
        shutil.copy(srcfile, dstfile)
    return

def generate_one_hot_samples_from_folders(src, dst, start, end):
    folder_num = end - start + 1
    for i in range(folder_num):
        csv_path = "%s/data%d/test.csv" % (src, start+i)
        data = pd.read_csv(csv_path)
        num_of_img = len(data)

        for index in range(num_of_img):
            img = data.iloc[index]['name'].strip()
            action = data.iloc[index]['action']
            actionPath = "%s/%d" % (dst, action)
            if not os.path.exists(actionPath):
                os.mkdir(actionPath)
            file = os.path.basename(img)    
            srcfile = "%s/data%d/img/%s" % (src, start+i, file)
            dstfile = actionPath + "/" + file
            shutil.move(srcfile, dstfile)
    return    

def merge_seq_4_images(srcList, output_file):
    img1 = Image.open(srcList[0])
    img2 = Image.open(srcList[1])
    img3 = Image.open(srcList[2])
    img4 = Image.open(srcList[3])

    w, h = img1.size[0], img1.size[1]
    merge_img = Image.new('RGB', (w * 2, h * 2), 0xffffff)
    merge_img.paste(img1, (0, 0))
    merge_img.paste(img2, (w, 0))
    merge_img.paste(img3, (0, h))
    merge_img.paste(img4, (w, h))
    merge_img.save(output_file)
    return

def generate_merge_seq_4_images(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    imgDst = dst + "/img"    
    if not os.path.exists(imgDst):
        os.mkdir(imgDst)

    sampleCSVFilePath = dst + "/test.csv"
    sampleCSVFile = open(sampleCSVFilePath, "w")
    sampleCSVWriter = csv.writer(sampleCSVFile)
    sampleCSVWriter.writerow(["action", "name", 'action_name'])
    sampleCSVFile.close()

    total = pd.read_csv(sampleCSVFilePath)
    # total = pd.DataFrame(columns=('name', 'action'))

    srcData = pd.read_csv(src)
    num_of_img = len(srcData)
    srcList = []
    for index in range(num_of_img):
        img = srcData.iloc[index]['name'].strip()
        action = srcData.iloc[index]['action']
        srcList.append(img)
        if len(srcList) < 4:
            continue

        output_file = "%s/%d_%d.jpg" % (imgDst,  index, action)
        merge_seq_4_images(srcList, output_file)

        # row = pd.DataFrame({"name":output_file, "action":action}, index=["0"])
        # total = total.append(row, ignore_index=True)

        row = pd.DataFrame({"name":"","action":""}, index=["0"])
        row.iloc[0]["name"] = output_file
        row.iloc[0]["action"] = action
        row.iloc[0]['action_name'] = DEATHMATCH_ACTION5_NAME[action]
        total = total.append(row, ignore_index=True)

        # pop head img and push new img
        srcList.pop(0)

    total.to_csv(sampleCSVFilePath, index=False, sep=',')
    return

def generate_samples_from_folders(start, end, src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    imgDst = dst + "/img"    
    if not os.path.exists(imgDst):
        os.mkdir(imgDst)

    sampleCSVFilePath = dst + "/test.csv"
    sampleCSVFile = open(sampleCSVFilePath, "w")
    sampleCSVWriter = csv.writer(sampleCSVFile)
    sampleCSVWriter.writerow(["action", "name"])
    sampleCSVFile.close()

    total = pd.read_csv(sampleCSVFilePath)

    for i in range(start, end):
        folder = "data%d" % (i)
        csvFilePath = src + "/" + folder + "/test.csv"
        data = pd.read_csv(csvFilePath)
        total = pd.concat([data, total])

        num_of_img = len(data)
        for index in range(num_of_img):
            img = data.iloc[index]['name'].strip()
            file = os.path.basename(img)
            srcfile = src + "/" + folder + "/img/" + file
            dstfile = imgDst + "/" + file
            # shutil.move(srcfile, dstfile)
            # shutil.copy(srcfile, dstfile)

    total.to_csv(sampleCSVFilePath, index=False, sep=',')
    return

def split_data_to_train_test(src, trainDst, testDst):
    if not os.path.exists(trainDst):
        os.mkdir(trainDst)
    imgTrainDst = trainDst + "/img"
    if not os.path.exists(imgTrainDst):
        os.mkdir(imgTrainDst)

    if not os.path.exists(testDst):
        os.mkdir(testDst)
    imgTestDst = testDst + "/img"
    if not os.path.exists(imgTestDst):
        os.mkdir(imgTestDst)

    srcData = pd.read_csv(src)
    num_of_img = len(srcData)
    train_num = 10 * 0.8

    sampleCSVFilePath1 = trainDst + "/test.csv"
    sampleCSVFile = open(sampleCSVFilePath1, "w")
    sampleCSVWriter = csv.writer(sampleCSVFile)
    sampleCSVWriter.writerow(["action", "name", "action_name", "friend", "enemy"])
    sampleCSVFile.close()

    totalTrain = pd.read_csv(sampleCSVFilePath1)

    sampleCSVFilePath2 = testDst + "/test.csv"
    sampleCSVFile = open(sampleCSVFilePath2, "w")
    sampleCSVWriter = csv.writer(sampleCSVFile)
    sampleCSVWriter.writerow(["action", "name", "action_name", "friend", "enemy"])
    sampleCSVFile.close()

    totalTest = pd.read_csv(sampleCSVFilePath2)

    counter = 0
    for index in range(num_of_img):
        img = srcData.iloc[index]['name'].strip()
        action = srcData.iloc[index]['action']
        friend = srcData.iloc[index]['Friends']
        enemy = srcData.iloc[index]['Enermy']
        file = os.path.basename(img)

        if counter < train_num:
            srcfile = img
            [shotname, extension] = os.path.splitext(file)
            dstfile = imgTrainDst + "/" + shotname + ".jpg"
            row = pd.DataFrame({"name":"","action":"","action_name":"","friend":"","enemy":""}, index=["0"])
            row.iloc[0]["name"] = dstfile
            row.iloc[0]["action"] = action
            row.iloc[0]['action_name'] = DEATHMATCH_ACTION5_NAME[action]
            row.iloc[0]['friend'] = friend
            row.iloc[0]['enemy'] = enemy
            totalTrain = totalTrain.append(row, ignore_index=True)
        else:
            srcfile = img
            [shotname, extension] = os.path.splitext(file)
            dstfile = imgTestDst + "/" + shotname + ".jpg"
            row = pd.DataFrame({"name":"","action":"","action_name":"","friend":"","enemy":""}, index=["0"])
            row.iloc[0]["name"] = dstfile
            row.iloc[0]["action"] = action
            row.iloc[0]['action_name'] = DEATHMATCH_ACTION5_NAME[action]
            row.iloc[0]['friend'] = friend
            row.iloc[0]['enemy'] = enemy            
            totalTest = totalTest.append(row, ignore_index=True)

        counter += 1

        # img_png = Image.open(srcfile)
        # img_jpg = img_png.resize((320, 180))
        # img_jpg.save(dstfile)

        # shutil.move(srcfile, dstfile)
        # shutil.copy(srcfile, dstfile)

        if counter == 10:
            counter = 0
    
    totalTrain.to_csv(sampleCSVFilePath1, index=False, sep=',')
    totalTest.to_csv(sampleCSVFilePath2, index=False, sep=',')
    return

def SupplementSamples(csvFile, action_num):
    total = pd.read_csv(csvFile)
    num_of_img = len(total)

    action_count = {}
    action_label = {}
    for i in range(action_num):
        action_count[i] = 0
        action_label[i] = []

    # Get the count of images of each label.
    for index in range(num_of_img):
        action = total.iloc[index]['action']
        name = total.iloc[index]['name']

        i = int(action)
        action_count[i] += 1
        action_label[i].append(name)

    max = 0
    action_max = {}
    for i in range(action_num):
        print("%s: %d" %(DEATHMATCH_ACTION5_NAME[i], action_count[i]))
        action_max[i] = action_count[i]
        if action_count[i] > max:
            max = action_count[i]

    for i in range(max):
        for action in range(action_num):
            if action_max[action] < max:
                new_index = random.randint(0, (action_count[action]-1))
                name = action_label[action][new_index]

                # generate new name
                [path, file] = os.path.split(name)
                [shotname, extension] = os.path.splitext(file)
                newName = "%s/%s_%d_%d%s" % (path, shotname, new_index, action_max[action], extension)
                shutil.copy(name, newName)

                row = pd.DataFrame({"name":"","action":"","action_name":""}, index=["0"])
                row.iloc[0]["name"] = newName
                row.iloc[0]["action"] = action
                row.iloc[0]['action_name'] = DEATHMATCH_ACTION5_NAME[action]
                total = total.append(row, ignore_index=True)

                action_max[action] += 1

    total.to_csv(csvFile, index=False, sep=',')
    return

def CutSamples(csvFile):
    total = pd.read_csv(csvFile)
    num_of_img = len(total)

    for index in range(num_of_img):
        action = total.iloc[index]['action']
        name = total.iloc[index]['name']

        img = Image.open(name)
        img2 = img.crop((0, 0, 256, 216))
        img2.save(name)
    return

def HorizontalFlipSamples(csvFile):
    total = pd.read_csv(csvFile)
    num_of_img = len(total)

    for index in range(num_of_img):
        action = total.iloc[index]['action']
        name = total.iloc[index]['name']

        bAdd = False
        if action == 3:
            new_action = 4
            bAdd = True
        elif action == 4:
            new_action = 3
            bAdd = True

        if bAdd:
            img = Image.open(name)
            img2 = img.transpose(Image.FLIP_LEFT_RIGHT)

            # generate new name
            [path, file] = os.path.split(name)
            [shotname, extension] = os.path.splitext(file)
            new_name = "%s/%s_%d%s" % (path, shotname, new_action, extension)
            img2.save(new_name)

            row = pd.DataFrame({"name":"","action":"","action_name":""}, index=["0"])
            row.iloc[0]["name"] = new_name
            row.iloc[0]["action"] = new_action
            row.iloc[0]['action_name'] = DEATHMATCH_ACTION5_NAME[new_action]
            total = total.append(row, ignore_index=True)

    total.to_csv(csvFile, index=False, sep=',')
    return

def ModifyRightSamples(csvFile):
    total = pd.read_csv(csvFile)
    num_of_img = len(total)

    for index in range(num_of_img):
        action = total.iloc[index]['action']
        name = total.iloc[index]['name']

        if action == 4:
            total.loc[index, 'action'] = 3
            total.loc[index, 'action_name'] = DEATHMATCH_ACTION5_NAME[3]

    total.to_csv(csvFile, index=False, sep=',')
    return

def RemoveNoAction(csvFile):
    total = pd.read_csv(csvFile)
    num_of_img = len(total)

    for index in range(num_of_img):
        action = total.iloc[index]['action']
        name = total.iloc[index]['name']

        if action == 6:
            os.remove(name)
    return

def ModifyNoActionToTurn(csvFile):
    total = pd.read_csv(csvFile)
    num_of_img = len(total)
    last_action = None
    delete_list = []

    for index in range(num_of_img):
        action = total.iloc[index]['action']
        name = total.iloc[index]['name']

        if action == 6:
            if last_action == 4 or last_action == 5:
                action = last_action
                action_name = DEATHMATCH_ACTION5_NAME[action]
                total['action'][index] = action
                total['action_name'][index] = action_name
            else:
                delete_list.append(index)            
            
        elif action != 6:
            last_action = action

    total.drop(total.index[[index]], inplace=True)
    total.to_csv(csvFile, index=False, sep=',')
    return

if __name__ == '__main__':
    # split_data_to_train_test("../CFM-Dataset/10000/test.csv", "../CFM-Dataset/10000/train", "../CFM-Dataset/10000/test")
    # generate_merge_seq_4_images("../CFM-Dataset/10000/test/test.csv", "../CFM-Dataset/10000/validate4Merge")    
    # generate_merge_seq_4_images("../CFM-Dataset/10000/train/test.csv", "../CFM-Dataset/10000/data4Merge")
    # generate_one_hot_samples("../CFM-Dataset/10000/validate4Merge/test.csv", "../CFM-Dataset/10000/one_hot_validate4Merge") 

    # generate_samples_4_merge(101, 195, "../ViZDoom-Dataset/ImitationData2", "../data200Skip2Merge4", 2)
    # generate_samples_4_merge(196, 200, "../ViZDoom-Dataset/ImitationData2", "../validate_data200Skip2Merge4", 2)

    # generate_merge_seq_4_images("./validate_data50Skip2/test.csv", "validate4Merge")    
    # generate_merge_seq_4_images("./data/test.csv", "./data4Merge")

    # generate_samples(1, 50, "../doom_data_1_53", "./data")
    # generate_samples(51, 53, "../doom_data_1_53", "./validate_data")
    # generate_one_hot_samples("./data/test.csv", "./one_hot_data")
    # generate_one_hot_samples("../ViZDoom-Dataset/validate_data100Skip2Merge4/test_new_action.csv", "../ViZDoom-Dataset/one_hot_validate_data100Skip2Merge4_newAction")
    # generate_one_hot_samples_from_folders("../doom_data_1_53", "./one_hot_data", 1, 48)
    # generate_one_hot_samples_from_folders("../doom_data_1_53", "./one_hot_validate_data", 49, 53)

    # CutSamples("../CFM-Dataset/10000/train/test_modify.csv")
    # CutSamples("../CFM-Dataset/10000/test/test.csv")

    # RemoveNoAction("../CFM-Dataset/10000/cfm_img_data.csv")

    # ModifyNoActionToTurn("../CFM-Dataset/40800-Action7/test/test.csv")
    # ModifyNoActionToTurn("../CFM-Dataset/40800-Action7/train/test.csv")

    pass
