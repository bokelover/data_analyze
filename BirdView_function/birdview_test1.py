"""
鳥瞰変換のコード(ファンクション)
18.11.16~
Author: BokeLover

ARGS:
    input: Original Image
    output: Bird-View Transform Image
"""

import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

def projection_transform(image):
    """
    座標を指定して射影変換するコード
    :param image: path and name
    :return: Projection Transform image
    """

    # 画像の読み込み
    img_src = cv.imread(image, 1)

    # 画像サイズの取得
    size = tuple(np.array([img_src.shape[1],img_src.shape[0]]))
    #print(size)
    # 移し替える元の画像入力値
    perspective1 = np.float32([[10, 10],
                               [1270, 10],
                               [1270, 950],
                               [10, 950]])
    # 鳥瞰変換後の座標指定
    perspective2 = np.float32([[100, 10],
                               [900, 10],
                               [600, 700],
                               [400, 700]])
    # 透視変換行列の作成
    psp_matrix = cv.getPerspectiveTransform(perspective1, perspective2)

    # 透視変換
    img_psp = cv.warpPerspective(img_src, psp_matrix, size)

    return img_psp

def bird_view_transform(img_src):
    """
    鳥瞰変換の関数。
    :param image:
    :return: Bird view transform
    """
    # 画像サイズの取得 (決め打ちにより凍結)
    # size = tuple(np.array([img_src.shape[1],img_src.shape[0]]))
    # パラメータ設定
    height, width, channels = 480, 640, 3

    HVC = 20 # 仮想カメラ中心高さ
    HC = 2.0  # 実カメラ中心高さ
    DVC = 12 # 仮想カメラ中心の位置（奥行き方向）
    f = 485 # 実カメラの焦点距離
    fp = f  # 仮想カメラの焦点距離
    kakudo = 1.0  # 俯角。以下シータにて使用
    theta = kakudo / 180 * math.pi  # 角度変換
    s = math.sin(theta)  # sin(θ)
    c = math.cos(theta)  # cos(θ)
    cx = 640/2   # 実画像の画像中心
    cy = 480/2   # 実画像のy方向
    cxp = 640/2   # 鳥瞰画像のx方向
    cyp = 480/2   # 鳥瞰画像のy方向

    # 出力画像の作成
    top_view = np.zeros((height+100, width+100, 3))

    for y in range(0, 639):
        for x in range(0, 800):
                for col in range(0, 3):
                    xOrg = x - cx
                    yOrg = -y + cy

                    oX = 0.5 + (HVC / HC) * (f / fp) * c * (s / c - (yOrg * HVC * s - fp * HC * c + fp * DVC * s) / (
                            fp * HC * s + HVC * yOrg * c + fp * DVC * c)) * xOrg
                    oY = 0.5 + f * (
                            (yOrg * HVC * s - fp * HC * c + fp * DVC * s) / (fp * HC * s + HVC * yOrg * c + fp * DVC * c))
                    oX = oX + cxp
                    oY = -oY + cyp

                    if oX < 0 or oX > width - 1 or oY < 0 or oY > height - 1 or x > width or y > height or x == 0 or y == 0:
                        continue

                    if int(oX) + 1 >= width or int(oY) + 1 >= height:
                        top_view[y, x, col] = img_src[int(oY), int(oX), col] / 255
                        continue
                    # bilinear TransForm
                    f11 = img_src[int(oY), int(oX), col]
                    f12 = img_src[int(oY) + 1, int(oX), col]
                    f21 = img_src[int(oY), int(oX) + 1, col]
                    f22 = img_src[int(oY) + 1, int(oX) + 1,col]

                    dx2 = int(oX) + 1 - oX
                    dx1 = oX - int(oX)

                    dy2 = int(oY) + 1 - oY
                    dy1 = oY - int(oY)
                    top_view[y, x,col] = (dy2 * (f11 * dx2 + f21 * dx1) + dy1 * (f12 * dx2 + f22 * dx1)) / 255

    return top_view

