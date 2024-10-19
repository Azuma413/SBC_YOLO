import asyncio
import websockets
import json
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

pc = RTCPeerConnection()
video_recorder = None

async def handle_video_track(track):
    # 受信した動画フレームを表示します
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")
        cv2.imshow("Received Video", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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

    # WebSocket経由でアンサーを送信
    await websocket.send(json.dumps({
        "sdp": local_description.sdp,
        "type": local_description.type
    }))

    # 受信したトラックを処理
    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            print("Video track received")
            await handle_video_track(track)

async def main():
    async with websockets.serve(websocket_handler, "0.0.0.0", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
