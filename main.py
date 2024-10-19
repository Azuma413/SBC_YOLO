import cv2
import time
from rknnpool import rknnPoolExecutor
from func import myFunc
# pip install aiortc
import aiortc
from aiortc import RTCPeerConnection, RTCSessionDescription
import websockets
import asyncio
import json

cap = cv2.VideoCapture('IMG_7202.MOV')
# cap = cv2.VideoCapture(0)
modelPath = "test.rknn"
TPEs = 6
pool = rknnPoolExecutor(
    rknnModel=modelPath,
    TPEs=TPEs,
    func=myFunc)
if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool
            exit(-1)
        pool.put(frame)

frames, loopTime, initTime = 0, time.time(), time.time()
while (cap.isOpened()):
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break
    pool.put(frame)
    frame, flag = pool.get()
    if flag == False:
        break
    cv2.imshow('test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frames % 30 == 0:
        print(30 / (time.time() - loopTime), "fps")
        loopTime = time.time()
cap.release()
cv2.destroyAllWindows()
pool.release()
