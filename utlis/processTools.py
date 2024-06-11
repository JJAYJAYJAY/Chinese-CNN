import cv2
import numpy as np


def preprocess_image(img_file):
    # 从文件存储读取图像
    filestr = img_file.read()
    # 转换字符串数据到numpy数组
    np_img = np.frombuffer(filestr, np.uint8)
    # 从numpy数组加载图像
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 应用二值化
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def horizontal_projection(binary):
    return np.sum(binary, axis=1)


def vertical_projection(binary):
    # 展示一下二维化的结果
    return np.sum(binary, axis=0)