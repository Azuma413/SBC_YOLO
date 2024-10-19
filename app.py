import streamlit as st
import cv2
# from rknnpool import rknnPoolExecutor
# from func import myFunc
import time
from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

st.title("YOLOv10物体認識デモ")

OBJ_THRESH = st.slider("しきい値1", min_value=0.0, max_value=1.0, step=0.01, value=0.25)
NMS_THRESH = st.slider("しきい値2", min_value=0.0, max_value=1.0, step=0.01, value=0.45)

modelPath = "test.rknn"
TPEs = 6

IMG_SIZE = (640, 640)

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    x = position
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = np.exp(y) / np.exp(y).sum(2, keepdims=True)
    acc_metrix = np.array(range(mc)).astype(np.float32).reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data):
    max_det, nc = 300, len(CLASSES)

    boxes, scores = [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        scores.append(input_data[pair_per_branch*i+1])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    scores = [sp_flatten(_v) for _v in scores]

    # NumPyの配列に変換し、次元を追加して結合します。
    boxes = np.expand_dims(np.concatenate(boxes, axis=0), axis=0)
    scores = np.expand_dims(np.concatenate(scores, axis=0), axis=0)

    # スコアの最大値を取得します。
    max_scores = np.amax(scores, axis=-1)
    index = np.argsort(-max_scores, axis=-1)[:, :max_det]

    # インデックスを使ってボックスとスコアを取得します。
    boxes = np.take_along_axis(boxes, index[..., np.newaxis].repeat(boxes.shape[-1], axis=-1), axis=1)
    scores = np.take_along_axis(scores, index[..., np.newaxis].repeat(scores.shape[-1], axis=-1), axis=1)

    # スコアとインデックスを再度計算します。
    flat_scores = scores.reshape(scores.shape[0], -1)
    score_index = np.argsort(-flat_scores, axis=-1)[:, :max_det]
    scores = np.take_along_axis(flat_scores, score_index, axis=-1)

    # ラベルとインデックスを計算します。
    labels = score_index % nc
    index = score_index // nc

    # ボックスをインデックスに基づいて取得します。
    boxes = np.take_along_axis(boxes, index[..., np.newaxis].repeat(boxes.shape[-1], axis=-1), axis=1)

    # 最終的な予測結果を結合します。
    preds = np.concatenate([boxes, scores[..., np.newaxis], labels[..., np.newaxis]], axis=-1)

    mask = preds[..., 4] > OBJ_THRESH

    preds = [p[mask[idx]] for idx, p in enumerate(preds)][0]
    boxes = preds[..., :4]
    scores =  preds[..., 4]
    classes = preds[..., 5].astype(np.int64)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

def myFunc(rknn_lite, src):
    img = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img, 0)
    outputs = rknn_lite.inference(inputs=[img], data_format=['nhwc'])
    boxes, classes, scores = post_process(outputs)
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * src.shape[1] / IMG_SIZE[1]
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * src.shape[0] / IMG_SIZE[0]
    if boxes is not None:
        draw(src, boxes, scores, classes)
    return src

def initRKNN(rknnModel="./rknnModel/yolov5s.rknn", id=0):
    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print(rknnModel, "\t\tdone")
    return rknn_lite


def initRKNNs(rknnModel="./rknnModel/yolov5s.rknn", TPEs=1):
    rknn_list = []
    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, i % 3))
    return rknn_list


class rknnPoolExecutor():
    def __init__(self, rknnModel, TPEs, func):
        self.TPEs = TPEs
        self.queue = Queue()
        self.rknnPool = initRKNNs(rknnModel, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0

    def put(self, frame):
        self.queue.put(self.pool.submit(self.func, self.rknnPool[self.num % self.TPEs], frame))
        self.num += 1

    def get(self):
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        return fut.result(), True

    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()

frame_window = st.image([])
fps_text = st.empty()

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.cap = cv2.VideoCapture('IMG_7202.MOV')
        # self.cap = cv2.VideoCapture(0)
        self.pool = rknnPoolExecutor(
            rknnModel=modelPath,
            TPEs=TPEs,
            func=myFunc
        )
        if self.cap.isOpened():
            for i in range(TPEs + 1):
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.release()
                    del self.pool
                    exit(-1)
                self.pool.put(frame)
        self.count = 0
        self.loopTime = time.time()
        self.initTime = time.time()
    def recv(self, frame):
        self.count += 1
        if self.count % 30 == 0:
            self.count = 0
            fps_text.text(f"{30 / (time.time() - self.loopTime):.2f}fps")
            self.loopTime = time.time()
        ret, img = self.cap.read()
        if not ret:
            st.error("Could not read frame")
            return frame
        self.pool.put(img)
        img, flag = self.pool.get()
        if flag == False:
            return frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def on_close(self):
        self.pool.release()
        self.cap.release()