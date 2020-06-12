# 3D-CNN-Autoencoder


Powered by [satolab](https://qiita.com/satolab)

## Overview

encoder-decoderモデルに3DCNNを組みこんだ，動画再構成モデルです．

This is a video reconstruction model that combines 3D-CNN and encoder-decoder model.


## Model
coming soon

## Results
- 5,000 itr(input,output)
coming soon

## Usage
- datasetのダウンロード
Download dataset
http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html
- lib.pyのParseGRU()内の初期化メソッド，dataset変数に，
上記のdatasetが格納されたdirを指定してください
(動画ファイルのままで問題ございません)
Please specify the dir in which the above dataset is stored.
(No problem as a video file.)


- main.pyで学習．logs/generated_videos/3dconvにサンプルが保存されます．
Learn with main.py.
The sample is saved in logs/generated_videos/3dconv.

## 参考文献 References
動画ファイルのロード部分および3DCNNモデル定義部参考
https://github.com/DLHacks/mocogan
