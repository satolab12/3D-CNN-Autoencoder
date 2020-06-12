# 3D-CNN-Autoencoder


Powered by [satolab](https://qiita.com/satolab)

## Overview

encoder-decoderモデルに3DCNNを組みこんだ，動画再構成モデルです．

This is a video reconstruction model that combines 3D-CNN and encoder-decoder model.


## Model

![40.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/583727/91b0ffde-30ef-6226-8e14-54a30e1e0f80.png)

## Results
- 4,000 itr(upper:input,output)
![real_itr4000_no1.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/583727/c13a8820-9eec-b012-9cde-7ffd4e99504b.png)
![recon_itr4000_no1.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/583727/a99a8df6-999e-b92a-bef7-070b59983081.png)

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
