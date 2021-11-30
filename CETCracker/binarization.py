import os
import cv2
import easyocr
import argparse

import ssl

#ssl._create_default_https_context = ssl._create_unverified_context

# img_dir_path = 'D:/dev/data/CET_VerifyPic'
source_dir = 'rawdata/test'
output_dir = 'data/test'

def binarization(dir, filename, save=False):
    input_img_file = os.path.join(dir, filename)
    image = cv2.imread(input_img_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # print("threshold value %s" % ret)  # 打印阈值，超过阈值显示为白色，低于该阈值显示为黑色
    # cv2.imshow("threshold", binary) #显示二值化图像
    # cv2.waitKey(0)
    if save:
        cv2.imwrite(os.path.join(output_dir, filename), binary)
    return binary

def ocr_test():
    cnt = 0
    correctCnt = 0
    reader = easyocr.Reader(['en'])
    for root, ds, fs in os.walk(source_dir):
        for f in fs:
            text = reader.readtext(binarization(source_dir, f))
            # text = reader.readtext(os.path.join(img_dir_path, f))
            if len(text) < 1:
                result = ''
            else:
                result = str(text[0][1]).lower().replace(' ', '')
            print(f + ':' + result)
            cnt += 1
            if f.split('.')[0] == result:
                correctCnt += 1
    print('accuracy:' + str(round(correctCnt * 100.0 / cnt, 2)) + '%')


def save_binarization(img_dir_path):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for base, dirs, filenames in os.walk(img_dir_path):
        for filename in filenames:
            binarization(img_dir_path, filename, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='small util to binarization images', prefix_chars='-')
    parser.add_argument('--source_dir', default='rawdata/train', type=str, required=True)
    parser.add_argument('--output_dir', default='data/train', type=str, required=True)
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir

    save_binarization(source_dir)
    # ocr_test()
    pass