import matplotlib.pyplot as plt
import sys, math
import time
import numpy as np
from skimage import io, color
"""
'19. 1. 21 BokeLover
SLIC: Simple Linear Iterative Clusteringを一から作る。
2010年にAchantaらによって発明された。
skimage内にも関数はある。処理は1分〜２分くらいかかる。Cython化必要がある。

SLICアルゴリズムは初期化と繰り返し処理の2構造
初期化処理関数：fit_init()
繰り返し処理関数:fit_iter()
とする。メイン関数はfit関数。

極簡単にまとめると以下のような流れのアルゴリズムであり、基本的にはk-meansクラスタリングのただのおばけ
1.rgb空間をlab空間へと射影
2.画素位置(x, y)にある色(l, a, b)を(l, a, b, x, y)空間へと射影
3.k個のクラスタ中心を等間隔に初期化
4.i番目の点と最寄りのクラスタ中心距離d[i]を∞遠方に初期化 (収束家のため)
5.クラスタ直径の近似値 S = sqrt(N/K) （∵N:画素数)を計算する。
"""

class SLIC:
    def __init__(self, k, m = 20):
        """
        コンストラクタ
        :param k: スーパーピクセルで分割する数
        :param m: スパースな各点の関連付重みパラメータ
                　論文では10〜40が推奨。デフォルトは20と定義
        """
        self.k = k
        self.m = m
        self.iter_max = 10 # 論文による。

    def fit(self, img_path):
        """
        スーパーピクセルの計算
        :return: マスク行列
        """
        self.fit_init(img_path)
        self.fit_iter()
        return self.l

    def fit_init(self, img_path):
        """
        初期化処理関数
        :param img_path:
        :return:
        """
        img_rgb = io.imread(img_path)
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise Exception("RGB形式でなく実行不能です。 画像構造は{}です。".format(img_rgb.shape))
        img_lab = color.rgb2lab(img_rgb)
        self.height = img_lab.shape[0]
        self.width = img_lab.shape[1]
        self.pixels = []
        for h in range(self.height):
            for w in range(self.width):
                self.pixels.append(np.array([img_lab[h][w][0],img_lab[h][w][1], img_lab[h][w][2], h, w]))
        self.size = len(self.pixels)
        # 標準化された空間におけるクラスター中心のInitialize
        self.cluster_center = []
        k_w = int(math.sqrt(self.k * self.width / self.height)) + 1
        k_h = int(math.sqrt(self.k * self.height / self.width)) + 1
        for h_cnt in range(k_h):
            h = (2 * h_cnt + 1) * self.height // (2 * k_h)
            for w_cnt in range(k_w):
                w = (2 * w_cnt + 1) * self.width // (2 * k_w)
                self.cluster_center.append(self.pixels[h * self.width + w])
        self.k = k_w * k_h
        self.l = [None] * self.size # クラスターラベル
        self.d = [math.inf] * self.size # ピクセル間の距離
        self.S = int(math.sqrt(self.size/self.k)) # クラスタ直径の近似値
        self.metric = np.diagflat([1 / (self.m**2)] * 3 + [1 / (self.S**2)]*2)

    def fit_iter(self):
        """
        繰り返し処理関数
        :return:
        """
        for iter_cnt in range(self.iter_max):
            for center_idx, center in enumerate(self.cluster_center):
                for h in range(max(0, int(center[3]) - self.S), min(self.height, int(center[3]) + self.S)):
                    for w in range(max(0, int(center[4]) - self.S), min(self.width, int(center[4]) + self.S)):
                        d = self.distance(self.pixels[h * self.width + w], center)

                        if d < self.d[h*self.width + w]:
                            self.d[h * self.width + w] = d
                            self.l[h * self.width + w] = center_idx

            self.calc_new_center()

    def distance(self, x, y):
        # L2ノルムのユークリッド距離を返します
        return (x-y).dot(self.metric).dot(x-y)

    def calc_new_center(self):
        # クラスター中心を再計算します。
        cnt = [0] * self.k
        new_cluster_center = [np.array([0., 0., 0., 0., 0.]) for _ in range(self.k)]
        for i in range(self.size):
            new_cluster_center[self.l[i]] += self.pixels[i]
            cnt[self.l[i]] += 1
        for i in range(self.k):
            new_cluster_center[i] /= cnt[i]
        self.cluster_center = new_cluster_center

    def transform(self):
        # 計算後、RGB空間に戻す射影関数
        cnt = [0] * self.k
        cluster_color = [np.array([0., 0., 0.]) for _ in range(self.k)]
        for i in range(self.size):
            cluster_color[self.l[i]] += self.pixels[i][:3]
            cnt[self.l[i]] += 1
        for i in range(self.k):
            cluster_color[i] /= cnt[i]

        # 画像を生成
        new_img_lab = np.zeros((self.height, self.width, 3))
        for h in range(self.height):
            for w in range(self.width):
                new_img_lab[h][w] = cluster_color[self.l[h * self.width + w]]
        return color.lab2rgb(new_img_lab)

# メイン関数
# t1, t2, elapssed_timeは処理時間計測用
t1 = time.time()

# slicデバッグ用コード
slic = SLIC(k = 200)
slic.fit("./frame00819.jpg")
res = slic.transform()
io.imshow(res)
# ここまで

t2 = time.time()
elapssed_time = t2-t1
print("SLIC処理に要した時間:{0:.2f}秒".format(elapssed_time))
io.show()