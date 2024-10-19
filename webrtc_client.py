import cv2
import time
from rknnpool import rknnPoolExecutor
from func import myFunc
import aiortc
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import websockets
import asyncio
import json

pc = RTCPeerConnection()
webcam = None

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, cap, pool):
        super().__init__()  # VideoStreamTrackの初期化
        self.cap = cap
        self.pool = pool

    async def recv(self):
        frame = await self.next_frame()
        return frame

    async def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.pool.put(frame)
        processed_frame, flag = self.pool.get()
        if not flag:
            return
        
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        return aiortc.VideoFrame.from_ndarray(frame_rgb, format="rgb24")

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

async def main():
    global webcam, pc
    cap = cv2.VideoCapture('IMG_7202.MOV')
    # cap = cv2.VideoCapture(0) # use webcam
    modelPath = "test.rknn"
    TPEs = 6
    pool = rknnPoolExecutor(
        rknnModel=modelPath,
        TPEs=TPEs,
        func=myFunc
    )
    
    if (cap.isOpened()):
        for i in range(TPEs + 1):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                del pool
                exit(-1)
            pool.put(frame)

        webcam = VideoTransformTrack(cap, pool)
        pc.addTrack(webcam)

        async with websockets.serve(websocket_handler, "0.0.0.0", 8765):
            await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
