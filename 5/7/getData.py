# coding=utf-8
# 代码来源： https://github.com/AlexiaJM/Deep-learning-with-cats/tree/master/Setting%20up%20the%20data
# 前两个zip，可以从这里下载：
# https://pan.baidu.com/share/link?shareid=2458390361&uk=2114378943&fid=183865443129559
# https://pan.baidu.com/share/link?shareid=3955394302&uk=2114378943&fid=716621184933841
# 后面的url直接从浏览器访问下载即可


import os
import urllib.request
import zipfile
import shutil
import glob
import cv2
import math

url1='https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/Data/CAT_DATASET_01.zip'
url2='https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/Data/CAT_DATASET_02.zip'
url3='https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/Data/00000003_015.jpg.cat'
curr_dir = os.path.dirname(__file__)
file1 = os.path.join(curr_dir,'data','CAT_DATASET_01.zip')
file2 = os.path.join(curr_dir,'data','CAT_DATASET_02.zip')
file3 = os.path.join(curr_dir,'data','00000003_015.jpg.cat')


def download(url,savefile):
    raise("下载不下来，自己看顶上的说明，来源", url, '保存为', savefile)
    urllib.request.urlretrieve(url, savefile)


def down():   
    if not os.path.exists(file1):
        print("start download",file1)
        download(url1,file1)
    if not os.path.exists(file2):
        print("start download",file2)
        download(url2,file2)
    if not os.path.exists(file3):
        print("start download",file3)
        download(url3,file3)    
    with zipfile.ZipFile(file1) as zf:        
        zf.extractall(os.path.join(curr_dir,"dataset"))
    with zipfile.ZipFile(file2) as zf:
        zf.extractall(os.path.join(curr_dir,"dataset"))
    for d in ['CAT_00','CAT_01','CAT_02','CAT_03','CAT_04','CAT_05','CAT_06']:
        for f in os.listdir(os.path.join(curr_dir,"dataset",d)):
            shutil.move(os.path.join(curr_dir,"dataset",d,f), \
                        os.path.join(curr_dir,"dataset",f))
        os.removedirs(os.path.join(curr_dir,"dataset",d))
    os.remove(os.path.join(curr_dir,"dataset","00000003_019.jpg.cat"))
    shutil.copy(file3,os.path.join(curr_dir,"dataset","00000003_015.jpg.cat"))
    delids=["00000004_007.jpg", "00000007_002.jpg", "00000045_028.jpg", "00000050_014.jpg", "00000056_013.jpg", 
            "00000059_002.jpg", "00000108_005.jpg", "00000122_023.jpg", "00000126_005.jpg", "00000132_018.jpg", 
            "00000142_024.jpg", "00000142_029.jpg", "00000143_003.jpg", "00000145_021.jpg", "00000166_021.jpg", 
            "00000169_021.jpg", "00000186_002.jpg", "00000202_022.jpg", "00000208_023.jpg", "00000210_003.jpg", 
            "00000229_005.jpg", "00000236_025.jpg", "00000249_016.jpg", "00000254_013.jpg", "00000260_019.jpg", 
            "00000261_029.jpg", "00000265_029.jpg", "00000271_020.jpg", "00000282_026.jpg", "00000316_004.jpg", 
            "00000352_014.jpg", "00000400_026.jpg", "00000406_006.jpg", "00000431_024.jpg", "00000443_027.jpg", 
            "00000502_015.jpg", "00000504_012.jpg", "00000510_019.jpg", "00000514_016.jpg", "00000514_008.jpg", 
            "00000515_021.jpg", "00000519_015.jpg", "00000522_016.jpg", "00000523_021.jpg", "00000529_005.jpg", 
            "00000556_022.jpg", "00000574_011.jpg", "00000581_018.jpg", "00000582_011.jpg", "00000588_016.jpg", 
            "00000588_019.jpg", "00000590_006.jpg", "00000592_018.jpg", "00000593_027.jpg", "00000617_013.jpg", 
            "00000618_016.jpg", "00000619_025.jpg", "00000622_019.jpg", "00000622_021.jpg", "00000630_007.jpg", 
            "00000645_016.jpg", "00000656_017.jpg", "00000659_000.jpg", "00000660_022.jpg", "00000660_029.jpg", 
            "00000661_016.jpg", "00000663_005.jpg", "00000672_027.jpg", "00000673_027.jpg", "00000675_023.jpg", 
            "00000692_006.jpg", "00000800_017.jpg", "00000805_004.jpg", "00000807_020.jpg", "00000823_010.jpg", 
            "00000824_010.jpg", "00000836_008.jpg", "00000843_021.jpg", "00000850_025.jpg", "00000862_017.jpg", 
            "00000864_007.jpg", "00000865_015.jpg", "00000870_007.jpg", "00000877_014.jpg", "00000882_013.jpg", 
            "00000887_028.jpg", "00000893_022.jpg", "00000907_013.jpg", "00000921_029.jpg", "00000929_022.jpg", 
            "00000934_006.jpg", "00000960_021.jpg", "00000976_004.jpg", "00000987_000.jpg", "00000993_009.jpg", 
            "00001006_014.jpg", "00001008_013.jpg", "00001012_019.jpg", "00001014_005.jpg", "00001020_017.jpg", 
            "00001039_008.jpg", "00001039_023.jpg", "00001048_029.jpg", "00001057_003.jpg", "00001068_005.jpg", 
            "00001113_015.jpg", "00001140_007.jpg", "00001157_029.jpg", "00001158_000.jpg", "00001167_007.jpg", 
            "00001184_007.jpg", "00001188_019.jpg", "00001204_027.jpg", "00001205_022.jpg", "00001219_005.jpg", 
            "00001243_010.jpg", "00001261_005.jpg", "00001270_028.jpg", "00001274_006.jpg", "00001293_015.jpg", 
            "00001312_021.jpg", "00001365_026.jpg", "00001372_006.jpg", "00001379_018.jpg", "00001388_024.jpg", 
            "00001389_026.jpg", "00001418_028.jpg", "00001425_012.jpg", "00001431_001.jpg", "00001456_018.jpg", 
            "00001458_003.jpg", "00001468_019.jpg", "00001475_009.jpg", "00001487_020.jpg"]
    for f in delids:
        os.remove(os.path.join(curr_dir,"dataset",f))
        os.remove(os.path.join(curr_dir,"dataset",f+".cat"))

def rotateCoords(coords, center, angleRadians):
    # Positive y is down so reverse the angle, too.
    angleRadians = -angleRadians
    xs, ys = coords[::2], coords[1::2]
    newCoords = []
    n = min(len(xs), len(ys))
    i = 0
    centerX = center[0]
    centerY = center[1]
    cosAngle = math.cos(angleRadians)
    sinAngle = math.sin(angleRadians)
    while i < n:
        xOffset = xs[i] - centerX
        yOffset = ys[i] - centerY
        newX = xOffset * cosAngle - yOffset * sinAngle + centerX
        newY = xOffset * sinAngle + yOffset * cosAngle + centerY
        newCoords += [newX, newY]
        i += 1
    return newCoords

def preprocessCatFace(coords, image):
    leftEyeX, leftEyeY = coords[0], coords[1]
    rightEyeX, rightEyeY = coords[2], coords[3]
    mouthX = coords[4]
    if leftEyeX > rightEyeX and leftEyeY < rightEyeY and \
            mouthX > rightEyeX:
        # The "right eye" is in the second quadrant of the face,
        # while the "left eye" is in the fourth quadrant (from the
        # viewer's perspective.) Swap the eyes' labels in order to
        # simplify the rotation logic.
        leftEyeX, rightEyeX = rightEyeX, leftEyeX
        leftEyeY, rightEyeY = rightEyeY, leftEyeY

    eyesCenter = (0.5 * (leftEyeX + rightEyeX), 0.5 * (leftEyeY + rightEyeY))

    eyesDeltaX = rightEyeX - leftEyeX
    eyesDeltaY = rightEyeY - leftEyeY
    eyesAngleRadians = math.atan2(eyesDeltaY, eyesDeltaX)
    eyesAngleDegrees = eyesAngleRadians * 180.0 / math.pi

    # Straighten the image and fill in gray for blank borders.
    rotation = cv2.getRotationMatrix2D(eyesCenter, eyesAngleDegrees, 1.0)
    imageSize = image.shape[1::-1]
    straight = cv2.warpAffine(image, rotation, imageSize, borderValue=(128, 128, 128))

    # Straighten the coordinates of the features.
    newCoords = rotateCoords(coords, eyesCenter, eyesAngleRadians)

    # Make the face as wide as the space between the ear bases.
    w = abs(newCoords[16] - newCoords[6])
    # Make the face square.
    h = w
    # Put the center point between the eyes at (0.5, 0.4) in
    # proportion to the entire face.
    minX = eyesCenter[0] - w/2
    if minX < 0:
        w += minX
        minX = 0
    minY = eyesCenter[1] - h*2/5
    if minY < 0:
        h += minY
        minY = 0

    # Crop the face.
    crop = straight[int(minY):int(minY+h), int(minX):int(minX+w)]
    # Return the crop.
    return crop

def catface():
    for img_filename in glob.glob(os.path.join(curr_dir,"dataset","*.jpg")):
        print(img_filename)
        coords_text = open(img_filename+".cat").readline().strip()
        coords = [int(i) for i in coords_text.split()[1:]]
        image = cv2.imread(img_filename)
        crop = preprocessCatFace(coords, image)
        if crop is None:
            print('Failed:', img_filename)
            continue
        h, w, colors = crop.shape
        if min(h,w) >= 64:
            Path1 = img_filename.replace("dataset","data64")
            cv2.imwrite(Path1, crop)
        if min(h,w) >= 128:
            Path2 = img_filename.replace("dataset","data128")
            cv2.imwrite(Path2, crop)

def main():
    # down()
    catface()

if __name__ == '__main__':
    main()