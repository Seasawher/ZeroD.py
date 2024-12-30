import sys
import os

sys.path.append(os.pardir)
import numpy as np
import pickle
import requests
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

def get_data():
    """テストデータを取得する"""
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    """学習済みの重みパラメータを読み込む"""
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    """学習済みニューラルネットワークと入力データが与えられたときに、推論される値を返す"""
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def download_sample_weight():
    """学習済みの重みパラメータをGitHubからダウンロードする"""
    if os.path.exists("sample_weight.pkl"):
        print("ファイルが既に存在するのでダウンロードをスキップします")
        return

    url = "https://github.com/oreilly-japan/deep-learning-from-scratch/raw/refs/heads/master/ch03/sample_weight.pkl"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open("sample_weight.pkl", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("ダウンロード完了")
    else:
        print(f"ダウンロードエラー: ステータスコード {response.status_code}")

if __name__ == '__main__':
    download_sample_weight()
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)

        # 最も確率が高い値を選ぶ
        p = np.argmax(y_batch, axis=1)

        # 予測を正解と比較して、合っていたらインクリメント
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
