import cv2
import aiortc
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import websockets
import asyncio
import json
from aiortc.contrib.media import VideoFrame
import numpy as np

pc = None
webcam = None

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap

    async def recv(self):
        await asyncio.sleep(0.01)
        frame = self.next_frame()
        if frame is None:
            return None
        pts, time_base = await self.next_timestamp()
        # フレームにタイムスタンプを設定
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # frameの平均と分散を表示
        print(np.mean(frame), np.var(frame))
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        return video_frame

async def run(pc, offer):
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return pc.localDescription

async def websocket_handler(websocket, path):
    global pc
    offer = await websocket.recv()
    offer = json.loads(offer)
    rtc_offer = RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])

    local_description = await run(pc, rtc_offer)

    await websocket.send(json.dumps({
        "sdp": local_description.sdp,
        "type": local_description.type
    }))
    async for message in websocket:
        candidate = json.loads(message)
        if 'candidate' in candidate:
            await pc.addIceCandidate(aiortc.RTCIceCandidate(candidate))

async def main():
    global webcam, pc
    pc = RTCPeerConnection()
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"ICE connection state is {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
    # cap = cv2.VideoCapture('IMG_7202.MOV')
    cap = cv2.VideoCapture(0) # use webcam
    if (cap.isOpened()):
        webcam = VideoTransformTrack(cap)
        pc.addTrack(webcam)
        async with websockets.serve(websocket_handler, "0.0.0.0", 8765):
            await asyncio.Future()
    else:
        print("Cannot open webcam")
    await pc.close()
    cap.release()

if __name__ == "__main__":
    asyncio.run(main())