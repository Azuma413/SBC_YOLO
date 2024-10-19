import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import av
from aiortc.contrib.media import MediaPlayer

st.title("My first Streamlit app")
st.write("Hello, world")

# スライダーで閾値を設定
threshold1_value = st.slider("Threshold1", min_value=0, max_value=1000, step=1, value=100)
threshold2_value = st.slider("Threshold2", min_value=0, max_value=1000, step=1, value=200)

# メディアプレーヤーを作成する関数
def create_player():
    # return MediaPlayer('/dev/video0', format='v4l2', options={'video_size': '640x480'}) # for Linux
    return MediaPlayer('IMG_7202.MOV')

# 映像フレームを処理するクラス
class VideoTransformer:
    def __init__(self):
        self.threshold1 = threshold1_value
        self.threshold2 = threshold2_value

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # エッジ検出などの映像処理を実行
        edges = cv2.Canny(img, self.threshold1, self.threshold2)
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# webrtc_streamerを設定
ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.RECVONLY,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    player_factory=create_player,
    video_processor_factory=VideoTransformer,
    async_processing=True
)

# VideoTransformerが存在する場合、閾値を更新
if ctx.video_transformer:
    ctx.video_transformer.threshold1 = threshold1_value
    ctx.video_transformer.threshold2 = threshold2_value