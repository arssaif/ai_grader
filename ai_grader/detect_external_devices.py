import os
import sys
import yolo_models
sys.modules['models'] = yolo_models
import shutil
import time
from pathlib import Path
from matplotlib import pyplot
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from utils.torch_utils import select_device
from yolo_models.experimental import attempt_load
from utils.general import (non_max_suppression, scale_coords, strip_optimizer, check_img_size)
import numpy as np

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

global imgsz
global model
global names
global device


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resizes and pads image while meeting stride-multiple constraints.

    Args:
        im (numpy.ndarray): Input image.
        new_shape (tuple or int): Target shape (height, width).
        color (tuple): Padding color.
        auto (bool): Minimum rectangle padding.
        scaleFill (bool): Stretch to fill.
        scaleup (bool): Allow scaling up.
        stride (int): Model stride.

    Returns:
        tuple: (processed image, ratio, (padding_width, padding_height))
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def draw_plot(filename, path, title):
    """
    Plots an image with a title and saves it to a specified path.

    Args:
        filename (numpy.ndarray): Image data to plot.
        path (str): Destination path to save the plot.
        title (str): Title for the plot.
    """
    pyplot.imshow(filename, cmap='gray')
    pyplot.title(title)
    ax = pyplot.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    pyplot.savefig(path, bbox_inches='tight')
    pyplot.clf()
    pyplot.cla()
    pyplot.close()


def detect_devices(adder, newName, user_id):
    """
    Detects external cardiac devices in an image using a pre-trained YOLO model.

    Args:
        adder (str): Relative path to the input image within the static directory.
        newName (str): Name for the output processed image.
        user_id (int/str): Identifier for the user to organize output directories.
    """
    pyplot.clf()
    pyplot.cla()
    pyplot.close()
    target = os.path.join(APP_ROOT, 'static/Patient_images')
    target = "/".join([target, adder])
    
    imgsz = 640
    device = select_device("")
    
    model = attempt_load('static/models/external_devices/external_devices.pt')  # load FP32 model
    model = model.to(device)
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    names = model.module.names if hasattr(model, 'module') else model.names
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    img = img.to(device)

    img_main = cv2.imread(target)
    img = letterbox(img_main, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)

    img = img.float()  # uint8 to fp16/32
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # pred
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.1, 0.1, classes=None, agnostic=False)
    title = ''
    results = []
    # Process detections
    for i, det in enumerate(pred):
        if det is None or len(det) <= 0:
            title = 'No External Devices'
        if det is not None and len(det):
            title = ' External Cardiac Devices '
            # Rescale boxes from img_size to img_main size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_main.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                results.append({
                    'xmin': int(xyxy[0]),
                    'ymin': int(xyxy[1]),
                    'xmax': int(xyxy[2]),
                    'ymax': int(xyxy[3]),
                    'class': names[int(cls)]
                })
        
        for bbox in results:
            cv2.rectangle(img_main, (bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax']), (255, 0, 0), 2)
            
    if not os.path.exists('static/external_devices/User' + str(user_id)):
        os.makedirs('static/external_devices/User' + str(user_id))
    path1 = 'static/external_devices/User' + str(user_id) + '/' + newName + '.jpg'
    draw_plot(img_main, path1, title)