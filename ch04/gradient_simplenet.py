import sys
import os
sys.path.append(os.pardir)

import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simplenet:
    """単純なニューラルネットワーク"""

    def W(self):
        """重みパラメータ"""
        return self.W

    def __init__(self):
        # ガウス分布で初期化
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        """予測を出す"""
        
        return np.dot(x, self.W)

    def loss(self, x, t):
        """損失関数の値を求める

        ただしxは入力でtは正解ラベル。
        """
        z = self.predict(x)
        # print(f"予測値: z={z}")
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss