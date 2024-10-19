import streamlit as st
import cv2
from rknnpool import rknnPoolExecutor
from func import myFunc
import time

st.title("My first Streamlit app")
st.write("Hello, world")

threshold1 = st.slider("Threshold1", min_value=0, max_value=1000, step=1, value=100)
threshold2 = st.slider("Threshold2", min_value=0, max_value=1000, step=1, value=200)

modelPath = "test.rknn"
TPEs = 6
pool = rknnPoolExecutor(
    rknnModel=modelPath,
    TPEs=TPEs,
    func=myFunc)

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('IMG_7202.MOV')

if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool
            exit(-1)
        pool.put(frame)

frames, loopTime, initTime = 0, time.time(), time.time()
frame_window = st.image([])

while (cap.isOpened()):
    frames += 1
    ret, frame = cap.read()
    if not ret:
        st.error("Could not read frame")
        break
    pool.put(frame)
    frame, flag = pool.get()
    if flag == False:
        break
    frame_window.image(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frames % 30 == 0:
        st.write(f"{30 / (time.time() - loopTime)} fps")
        loopTime = time.time()

cap.release()