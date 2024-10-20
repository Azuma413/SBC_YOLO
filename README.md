# 環境
SBC: radxa rock 5c lite 4GB
# セットアップ
### PC
[こちら](https://github.com/radxa-build/rock-5c/releases/)から`rock-5c_bookworm_cli`をダウンロードしてrufusでmicroSDに書き込む。

SBCにディスプレイ，電源，キーボードを繋ぐ。
### SBC
ログインは\
アカウント名：radxa\
パスワード：radxa

nmtuiでwifiに接続する。
```
setxkbmap us dvorak
sudo apt update && sudo apt upgrade -y
``` 
sudo apt install console-setup
sudo loadkeys dvorak
```
sudo visudo
```
以下の内容を追加する。
```
%sudo ALL=(ALL:ALL) ALL
radxa ALL=(ALL) NOPASSWD:ALL  ←追加
```
mDNSを設定
```
sudo nano /etc/avahi/services/ssh.service
```
以下の内容を書き込む。
```xml
<?xml version="1.0" standalone="no"?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name replace-wildcards="yes">%h</name>
  <service>
    <type>_ssh._tcp</type>
    <port>22</port>
  </service>
</service-group>
```
ssh接続できるようにする。
```
sudo systemctl restart avahi-daemon
sudo systemctl status avahi-daemon
sudo systemctl start ssh
```
### PC
(.ssh/known_hostsのrock-5cの部分を削除して)ssh接続する
```
ssh radxa@rock-5c.local
```
各種ダウンロード
```
sudo apt update
sudo apt install git python3-rknnlite2 rknn-model-zoo-rk3588 guvcview -y
mkdir SourceCode && cd SourceCode
git clone https://github.com/Azuma413/SBC_YOLO.git
sudo pip install aiortc websockets opencv-python streamlit==1.33.0 streamlit-webrtc --break-system-packages
```
動作テスト
```
cd SBC_YOLO
streamlit run app.py --server.headless true
```
# YOLOv10
### ファインチューニング
wslにリポジトリをダウンロード
```
git clone --recurse-submodules -j8 https://github.com/Azuma413/SBC_YOLO.git
```
python3.8のconda環境を作成
```
conda create -n rknn python=3.8
conda activate rknn
pip install ultralytics
cd SBC_YOLO/rknn-toolkit2/rknn-toolkit2
pip install -r packages/requirements_cp38-1.6.0.txt
pip install packages/rknn_toolkit2-2.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```
yolo用のconda環境を作成
```
conda create -n yolov10 python=3.9
conda activate yolov10
cd yolov10
pip install -r requirements.txt
pip install -e .
pip install "numpy<2"
```

TODO: ファインチューニングの具体的な方法を書く。
yolo detect train data=coco.yaml model=yolov10n/s/m/b/l/x.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7
ファインチューニングの方法については[こちら](https://qiita.com/hirekatsu0523/items/f2f0e1a0f8a9ea92d913)を参照。

### rknn変換
yoloは通常，学習段階で640*640に最適化されているので，いじらない方が良い。
```
conda activate yolov10
yolo export model=path/to/your_model.pt format=rknn opset=13 simplify imgsz=640
```
これでrknnに対応した形式のonnxファイルが出力される。
```
conda activate rknn
cp your_model.onnx rknn_model_zoo/examples/yolov10/python
cd rknn_model_zoo/examples/yolov10/python
python convert.py your_model.onnx rk3588 i8 your_model.rknn
cd ../../../..
cp rknn_model_zoo/examples/yolov10/python/your_model.rknn .
```


# リポジトリ
[rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)\
[rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)

# 参考
[公式ドキュメント](https://developer.d-robotics.cc/rdk_doc/en/Basic_Development)\
[転倒データセット](https://ieeexplore.ieee.org/document/9171857/algorithms?tabFilter=dataset#algorithms)\
[転倒データセット2](https://universe.roboflow.com/hero-d6kgf/yolov5-fall-detection)\
[転倒データセット3](https://www.perplexity.ai/search/zhuan-dao-jian-zhi-shou-fa-tot-YMOwBnkGTA69gU3SQhbZMw)\
[NPUで高速推論](https://qiita.com/ysuito/items/a0d3201581f9057c973b#npu%E3%81%A8%E3%81%AF)\
[YOLOv8で転倒検出](https://github.com/pahaht/YOLOv8-Fall-detection)