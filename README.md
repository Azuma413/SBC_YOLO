# 環境
SBC: radxa rock 5c lite 4GB
# セットアップ
### PC
[こちら](https://github.com/radxa-build/rock-5c/releases/)から`rock-5c_bookworm_cli`をダウンロードしてrufusでmicroSDに書き込む。

SBCにディスプレイ，電源，キーボードを繋ぐ。
### SBC
ログインは
アカウント名：radxa
パスワード：radxa

nmtuiでwifiに接続する。
```
sudo apt update && sudo apt upgrade -y
sudo apt install console-setup
sudo loadkeys dvorak
```
エラーが発生しても気にしない。
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
vscodeのリモートエクスプローラを開く
```
ssh radxa@rock-5c.local
```
# 動作確認
YOLOv8を動かしてみる。

ファインチューニングの方法については[こちら](https://qiita.com/hirekatsu0523/items/f2f0e1a0f8a9ea92d913)を参照。



# SBC_YOLO
### 実装したい機能
- 高精度な転倒検出\
前提として，メモリ消費量は少なければ少ないほど，安いSBCで動作可能。
- 年齢推定？
- 人物の判別はやりたい
- 人物ごとの移動距離等の時系列データの保存\
→webで閲覧可能

SOTA手法とかもあると思うけれど，サービスの継続性とかを考えると，結局YOLOがよさそう。


# 参考
[公式ドキュメント](https://developer.d-robotics.cc/rdk_doc/en/Basic_Development)

[転倒データセット](https://ieeexplore.ieee.org/document/9171857/algorithms?tabFilter=dataset#algorithms)

[NPUで高速推論](https://qiita.com/ysuito/items/a0d3201581f9057c973b#npu%E3%81%A8%E3%81%AF)

[YOLOv8で転倒検出](https://github.com/pahaht/YOLOv8-Fall-detection)

[Google検索](https://www.google.com/search?q=yolo+human+falling&sca_esv=fef8a0a8565c2553&sxsrf=ADLYWIK8XVuBc0kY8tuDLcxB5Fnie2qsaA%3A1728566423082&source=hp&ei=l9QHZ_rhAYfCvr0Pl7ekwQg&iflsig=AL9hbdgAAAAAZwfipxpoqxLX23iIPpkSSvkbhSfRzRyT&ved=0ahUKEwi6ifbc84OJAxUHoa8BHZcbKYgQ4dUDCA8&uact=5&oq=yolo+human+falling&gs_lp=Egdnd3Mtd2l6IhJ5b2xvIGh1bWFuIGZhbGxpbmcyCBAAGIAEGKIEMggQABiABBiiBDIIEAAYgAQYogQyCBAAGIAEGKIEMggQABiABBiiBEjyHlCjAViaHXABeACQAQCYAXSgAbINqgEEMTQuNLgBA8gBAPgBAZgCE6ACzA2oAgrCAgcQIxgnGOoCwgIMECMYgAQYExgnGIoFwgINEAAYgAQYsQMYgwEYBMICCxAAGIAEGLEDGIMBwgIQEAAYgAQYsQMYQxiDARiKBcICDhAAGIAEGLEDGIMBGIoFwgIKEAAYgAQYsQMYBMICBxAAGIAEGATCAgoQABiABBhDGIoFwgIFEAAYgATCAggQABiABBixA8ICCRAAGIAEGAQYCsICCBAAGIAEGMsBwgIEEAAYHsICBhAAGAgYHsICCBAAGAgYChgewgIKEAAYCBgKGA0YHsICBRAhGKABmAMEkgcEMTEuOKAHjC8&sclient=gws-wiz)\
色々ある。