#!/usr/bin/python
# -*- coding: utf-8 -*-

#
#  File Name	: 'dewarp.py'
#  Author	: Steve NGUYEN
#  Contact      : steve.nguyen.000@gmail.com
#  Created	: vendredi, novembre 18 2016
#  Revised	:
#  Version	:
#  Target MCU	:
#
#  This code is distributed under the GNU Public License
# 		which can be found at http://www.gnu.org/licenses/gpl.txt
#
#
#  Notes:	notes
#

import sys
import cv2
import numpy as np

bigmama = 'big_mama.xml'

map1 = np.asarray(
    cv2.cv.Load(bigmama, cv2.cv.CreateMemStorage(), 'map1'))
map2 = np.asarray(
    cv2.cv.Load(bigmama, cv2.cv.CreateMemStorage(), 'map2'))


image = cv2.imread(sys.argv[1])
image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)

dst = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
cv2.imshow("img", dst)
cv2.imwrite('dewarped.png', dst)
cv2.waitKey()
