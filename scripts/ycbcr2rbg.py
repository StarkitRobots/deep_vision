#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import cv2
# import numpy as np
import glob
import os

files = glob.glob(sys.argv[1] + '*.png')

if not os.path.exists(sys.argv[1] + '/rgb/'):
    os.makedirs(sys.argv[1] + '/rgb/')

for f in files:

    image = cv2.imread(f)
    image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)
    cv2.imshow("img", image)
    # print os.path.dirname(f), os.path.basename(f)
    print os.path.dirname(f) + '/rgb/' + os.path.basename(f)
    cv2.imwrite(os.path.dirname(f) + '/rgb/' + os.path.basename(f), image)
    # cv2.waitKey()
