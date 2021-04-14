#!encoding=utf-8
import os

formats=[".jpg",".png", ".jpeg",".bmp"]

def IsSubString(SubStrList, Str):
    flag = True
    for substr in SubStrList:
        if not (substr in Str):
            flag = False

    return flag


# 扫面文件
def GetFileList(FindPath, FlagStr=[]):
    FileList = []
    FileNames = os.listdir(FindPath)
    if len(FileNames) > 0:
        for fn in FileNames:
            if len(FlagStr) > 0:
                if IsSubString(FlagStr, fn):
                    fullfilename = os.path.join(FindPath, fn)
                    FileList.append(fullfilename)
            else:
                fullfilename = os.path.join(FindPath, fn)
                FileList.append(fullfilename)

    if len(FileList) > 0:
        FileList.sort()

    return FileList

def genImgList(data_root,cat_ids,f):
    dirs = os.listdir(data_root)
    for dir in dirs:
        classID = cat_ids[dir]
        classdir = os.path.join(data_root, dir)
        filelist = GetFileList(classdir)
        for img in filelist:
            if os.path.splitext(img)[1] not in formats:
                continue
            line = img + ' ' + str(classID) + '\n'  # 用空格代替转义字符 \t
            f.writelines(line)

train_txt = open('trainval.txt', 'w')
test_txt = open('val.txt', 'w')

# classes name list
classNames=["neg","pos"]
cat_ids = {v: i for i, v in enumerate(classNames)}
#class data root in Dataset
train_root="maskdata/train"
test_root="maskdata/val"

genImgList(train_root,cat_ids,train_txt)
genImgList(test_root,cat_ids,test_txt)


train_txt.close()
test_txt.close()

print("complete ! ! !")