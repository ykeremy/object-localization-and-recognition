# source tutorial: https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/
# opencv doc: https://docs.opencv.org/3.4/d6/d6d/classcv_1_1ximgproc_1_1segmentation_1_1SelectiveSearchSegmentation.html
#%% Imports
import os
import cv2 as cv
from SelectiveSearch import SelectiveSearch
import numpy as np

#%% Get all test image filenames
test_img_path = "./data/test/images"
filenames = os.listdir(test_img_path)

#%% Read all images
images = []
for fname in filenames:
    img_path = os.path.join(test_img_path, fname)
    images.append(cv.imread(img_path))

#%%
ss = SelectiveSearch()
boxes = ss.process([images[0]])
print(boxes[0].shape)

ss.set_width_threshold(60)
ss.set_height_threshold(60)
boxes = ss.process([images[0]])
print(boxes[0].shape)

# %% from tutorial, show bounding boxes on top of image
""" 
# number of region proposals to show
numShowRects = 100
# increment to increase/decrease total number
# of reason proposals to be shown
increment = 50

while True:
    # create a copy of original image
    imOut = image.copy()

    # itereate over all the region proposals
    for i, rect in enumerate(boxes):
        # draw rectangle for region proposal till numShowRects
        if (i < numShowRects):
            x, y, w, h = rect
            cv.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)
        else:
            break

    # show output
    cv.imshow("Output", imOut)

    # record key press
    k = cv.waitKey(0) & 0xFF

    # m is pressed
    if k == 109:
        # increase total number of rectangles to show by increment
        numShowRects += increment
    # l is pressed
    elif k == 108 and numShowRects > increment:
        # decrease total number of rectangles to show by increment
        numShowRects -= increment
    # q is pressed
    elif k == 113:
        break
# close image show window
cv.destroyAllWindows()
 """
