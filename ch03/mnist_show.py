import sys
import os

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    """画像を表示させる"""
    # PIL 用のデータオブジェクトに変換する
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(f'label: {label}')

print(f'img.shape: {img.shape}')

# 画像を元のサイズに変形する
img = img.reshape(28, 28)
print(f'img.shape: {img.shape}')

img_show(img)