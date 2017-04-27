#!/usr/bin/python
# -*- coding: utf-8 -*-

#
#  File Name	: 'add_noise.py'
#  Author	: Steve NGUYEN
#  Contact      : steve.nguyen.000@gmail.com
#  Created	: samedi, mars 11 2017
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

import os
import sys
import shutil
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import glob


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'USAGE: python add_noise.py DATAPATH/'
        print 'output new data in DATAPATH/generated'
        sys.exit()

    base = os.path.dirname(sys.argv[1])

    outname = '/generated/'
    files = glob.glob(sys.argv[1] + '*.png')

    if not os.path.exists(sys.argv[1] + outname):
        os.makedirs(sys.argv[1] + outname)

    dirname = base + outname

    for image in files:
        print image
        img = cv2.imread(image)

        # symmetry
        rimg = cv2.flip(img, 1)
        cv2.imshow('im', img)
        cv2.imshow('rim', rimg)
        base + outname + '' + image
        # cv2.imwrite(base + outname + '' + image)

        name = os.path.basename(image)
        # print 'test', dirname + name
        # print 'test', dirname + 'sym_' + name

        cv2.imwrite(dirname + name, img)
        cv2.imwrite(dirname + 'sym_' + name, rimg)

        for gs in [5]:
             # [3, 6]:  # size of the gaussian kernel

            bimg = cv2.blur(img, (gs, gs))
            brimg = cv2.blur(rimg, (gs, gs))

            cv2.imshow('bim', bimg)
            cv2.imshow('brim', brimg)

            # print 'test', dirname + str(gs) + '_' + name
            # print 'test', dirname + str(gs) + '_sym_' + name

            cv2.imwrite(dirname + str(gs) + '_' + name, bimg)
            cv2.imwrite(dirname + str(gs) + '_sym_' + name, brimg)

            # cv2.waitKey(0)
