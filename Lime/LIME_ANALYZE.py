import sys
sys.path.append('./tf-models/slim')
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib.pyplot as plt
import numpy as np
import os
from nets import inception
from preprocessing import inception_preprocessing
from datasets import imagenet
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image, ImageDraw, ImageFilter

"""
LIMEを用いたSuperPixel上で影響が大きい局所的な領域を解析する簡易特徴領域抽出機
使い方： 画像を入力として与える
　　　　初回の結果でTop5がPrintされる。その中から解析したいラベル(数字)をコンソールに書き込むだけ。
出力： 入力画像におけるラベルの影響が大きかった領域
モデル： Inception v3
スーパーピクセル： LIME準拠(sklearn quickshift)
"""
# 定数の設定 5が推奨
num_feature = 5
# 読み込みたい画像をここに入れる
image_path = './test.JPG'

np.set_printoptions(threshold=np.inf)
session = tf.Session()
image_size = inception.inception_v3.default_image_size

def transform_img_fn(path_list):
    out = []
    for f in path_list:
        image_raw = tf.image.decode_jpeg(open(f, 'rb').read(), channels=3)
        image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
        out.append(image)
    return session.run([out])[0]


def transform_img_to_tnsl(image):
    #新規追加： 画像Openではなく中間の処理
    out = []
    out.append(image)
    return session.run([out])[0]

def predict_fn(images):
    return session.run(probabilities, feed_dict={processed_images: images})

def reverse_mask(f_image, mask, reverse=False):
    """
    :param f_image: 最初の入力画像を入れる。大きさは(299,299,3)
    :param mask: LIMEによって得られたマスク
    :param reverse: Trueであれば、マスク処理が逆転する
    :return out_image: mask処理されたnp.array(299, 299, 3)
    """
    # サイズを取得
    width, height, channel = f_image.shape
    # 新しい画像を作成 s_image
    out_image = np.zeros((width, height, channel))

    # maskを参照し、1の領域を通過、 0領域をカットオフする（本来とは逆にしている
    for i in range(width):
        for j in range(height):
            if reverse:
                if mask[i][j] == 1:
                    # 1 の場合
                    pass
                elif mask[i][j] == 0:
                    # 0 の場合
                    out_image[i][j][0] = f_image[i][j][0]
                    out_image[i][j][1] = f_image[i][j][1]
                    out_image[i][j][2] = f_image[i][j][2]
            else:
                if mask[i][j] == 0:
                    # 1 の場合
                    pass
                elif mask[i][j] == 1:
                    # 0 の場合
                    out_image[i][j][0] = f_image[i][j][0]
                    out_image[i][j][1] = f_image[i][j][1]
                    out_image[i][j][2] = f_image[i][j][2]
    return out_image

def mask_combine(post_mask, pre_mask):
    """
    :param post_mask, pre_mask: マージするマスク
    :return: マージしたマスク
    """
    # サイズを取得
    width, height = post_mask.shape
    out_mask = np.zeros((width,height))
    for i in range(width):
        for j in range(height):
                if post_mask[i][j] == 0 and  pre_mask[i][j] == 0:
                    out_mask[i][j] = 0
                else:
                    out_mask[i][j] = 1
    return out_mask

names = imagenet.create_readable_names_for_imagenet_labels()
processed_images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
probabilities = tf.nn.softmax(logits)

checkpoints_dir = './tf-models/slim/pretrained'
init_fn = slim.assign_from_checkpoint_fn(
    os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
    slim.get_model_variables('InceptionV3'))
init_fn(session)

images = transform_img_fn([image_path])
preds = predict_fn(images)
print('###--- First Inception Result --- ###')
for x in preds.argsort()[0][-5:]:
    print(x, names[x], preds[0, x])
    # 逆順なので最後のxが最も良いラベル
    tmp_x = x
    tmp_prob = preds[0, x]

# 初回処理
f_image = images[0]
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(f_image, predict_fn, top_labels=5, hide_color=0, num_samples=1000)
f_temp, f_mask = explanation.get_image_and_mask(tmp_x, positive_only=True, num_features=num_feature, hide_rest=True)

# 入力処理
input_strings = input('Label No. is:' )
assert str.isdecimal(input_strings) is True, 'Error, please input Label No.'
target_label = int(input_strings)

# ループ処理
flag = 1
count = 1
while flag == 1:
    # Initialize
    print('\n #########   -----{} LOOP-----   #########'.format(count))

    flag = 0
    # Reverse Mask
    s_image = reverse_mask(f_image=f_image, mask=f_mask, reverse=True)

    cv_s_image = tf.image.convert_image_dtype(s_image, dtype=tf.float32)
    s_images = transform_img_to_tnsl(cv_s_image)

    r_img = s_images[0]
    preds2 = predict_fn(s_images)
    for x in preds2.argsort()[0][-5:]:
        print(x, names[x], preds2[0, x])
        tmp_x = x
        tmp_prob = preds[0, x]

        # flag処理
        if flag == 1:
            pass
        else:
            if target_label == tmp_x:
                flag = 1

    if flag == 0:
        # breakしないとエラー
        print('---   Finished!  ---\n')
        break

    explanation = explainer.explain_instance(r_img, predict_fn, top_labels=5, hide_color=0, num_samples=1000)
    t_temp, t_mask = explanation.get_image_and_mask(tmp_x, positive_only=True, num_features=num_feature, hide_rest=True)

    # maskの結合処理
    f_mask = mask_combine(f_mask, t_mask)
    count += 1

# 出力結果
print('label No.{} Explain Result'.format(target_label))
l_image = reverse_mask(f_image=f_image, mask=f_mask, reverse=False)
cv_l_image = tf.image.convert_image_dtype(l_image, dtype=tf.float32)
# inception_v3用フォーマットへの変換
l_images = transform_img_to_tnsl(cv_l_image)

preds2 = predict_fn(l_images)
for x in preds2.argsort()[0][-5:]:
    print(x, names[x], preds2[0, x])
    # 逆順なので最後のxが最も良いラベル
    tmp_x = x
    tmp_prob = preds[0, x]

plt.imshow(l_image / 2 + 0.5)
plt.tick_params (labelbottom = False,
                 labelleft = False)
plt.show()
