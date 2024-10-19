import os
import cv2
import sys
import argparse
from rknnlite.api import RKNNLite
from rknn_model_zoo.py_utils.coco_utils import COCO_test_helper
import numpy as np


OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

# IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)
IMG_SIZE = (1088, 608)

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

def post_process_yolov10(input_data):
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
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def setup_model(args):
    model_path = args.model_path
    platform = 'rknn'
    from rknn_model_zoo.py_utils.rknn_executor import RKNN_model_container 
    model = RKNN_model_container(args.model_path, args.target, args.device_id)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, required= True, help='model path, could be .pt or .rknn file')
    # parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
    # parser.add_argument('--device_id', type=str, default=None, help='device id')
    # data params
    # parser.add_argument('--anno_json', type=str, default='../../../datasets/COCO/annotations/instances_val2017.json', help='coco annotation path')
    # coco val folder: '../../../datasets/COCO//val2017'
    # parser.add_argument('--img_folder', type=str, default='../model', help='img folder path')
    # parser.add_argument('--coco_map_test', action='store_true', help='enable coco map test')
    args = parser.parse_args()
    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(args.model_path)
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    img = cv2.imread('bus.jpg')
    # video = cv2.VideoCapture('IMG_7202.MOV')
    # img_list = []
    # while True:
    #     # フレームを取得
    #     ret, frame = video.read()
    #     if not ret:
    #         break
    #     img_list.append(frame)
    # video.release()
    co_helper = COCO_test_helper(enable_letter_box=True)

    # output_list = []
    # run test
    # for img in img_list:
        # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
    pad_color = (0,0,0)
    img = co_helper.letter_box(im= img.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 736, 1280, 3 -> 1, 736, 1280, 3
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    outputs = rknn_lite.inference(inputs=[img], data_format=['nhwc'])
    # print(outputs.shape)
    boxes, classes, scores = post_process_yolov10(outputs)
    if boxes is not None:
        draw(img, co_helper.get_real_box(boxes), scores, classes)
    # output_list.append(img)
    # 動画として保存 mp4
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # video = cv2.VideoWriter('output.mp4', fourcc, 30, (1280, 736))
    
    # imgを保存
    cv2.imwrite('output.jpg', img[0])
