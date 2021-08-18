import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio


# Defining a function that will do the detections
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    