#!encoding=utf-8
import os

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


train_txt = open('trainval.txt', 'w')
test_txt = open('val.txt', 'w')
# 制作图片目录文件

# imgfile1 = GetFileList('trains/CCM')
# for img in imgfile1:
#     str1 = img+' '+'0'+'\n'  # 用空格代替转义字符 \t
#     train_txt.writelines(str1)

imgfile10 = GetFileList('maskdata/game_gauzeMask_data/train/neg')
for img in imgfile10:
    if os.path.splitext(img)[1]!='.jpg':
        continue
    str1 = img+' '+'0'+'\n'  # 用空格代替转义字符 \t
    train_txt.writelines(str1)

imgfile20 = GetFileList('maskdata/game_gauzeMask_data/train/pos')
for img in imgfile20:
    if os.path.splitext(img)[1]!='.jpg':
        continue
    str2 = img+' '+'1'+'\n'
    train_txt.writelines(str2)

# imgfile30 = GetFileList('trains/prison_samples/policemen')
# for img in imgfile30:
#     str3= img+' '+'2'+'\n'
#     train_txt.writelines(str3)

train_txt.close()

# 测试集文件列表
# imgfile = GetFileList('vals/kid')  # 将数据集放在与.py文件相同目录下
# for img in imgfile:
#     str5 = img + ' ' + '0' + '\n'
#     test_txt.writelines(str5)
#
# imgfile = GetFileList('vals/teacher')
# for img in imgfile:
#     str6= img + ' ' + '1' + '\n'
#     test_txt.writelines(str6)

test_txt.close()

print("成功生成文件列表")