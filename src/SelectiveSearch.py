import cv2 as cv
import numpy as np


class SelectiveSearch:
    def __init__(self):
        self.threshold = [0, 0]
        cv.setUseOptimized(True)
        cv.setNumThreads(8)

    def process(self, image_list):
        all_boxes = []
        for image in image_list:
            ss_instance = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss_instance.setBaseImage(image)
            ss_instance.switchToSelectiveSearchFast()

            # returns x, y, width, height form bounding boxes
            boxes = ss_instance.process()

            # filter according to width and height thresholds
            w_thresh, h_thresh = self.threshold
            boxes = boxes[np.where((boxes[:, 2] > w_thresh) * (boxes[:, 3] > h_thresh))]

            # convert bounding boxes to x1, y1, x2, y2 form
            # by setting x2 = x1 + width, y2 = y1 + height
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

            all_boxes.append(boxes)
        return all_boxes

    def set_width_threshold(self, x):
        self.threshold[0] = x

    def set_height_threshold(self, y):
        self.threshold[1] = y
