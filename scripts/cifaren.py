#!/usr/bin/python
# -*- coding: utf-8 -*-

#
#  File Name	: 'cifaren.py'
#  Author	: Steve NGUYEN
#  Contact      : steve.nguyen.000@gmail.com
#  Created	: lundi, novembre 21 2016
#  Revised	:
#  Version	:
#  Target MCU	:
#
#  This code is distributed under the GNU Public License
# 		which can be found at http://www.gnu.org/licenses/gpl.txt
#
#
#  Create a cifar like database
# the first byte is the label of the first image, which is a number in the
# range 0-9. The next 3072 bytes are the values of the pixels of the
# image. The first 1024 bytes are the red channel values, the next 1024
# the green, and the final 1024 the blue. The values are stored in
# row-major order, so the first 32 bytes are the red channel values of the
# first row of the image.

#

import sys
import numpy as np
import cv2
import os


def to_bin(filename, label, w=32, h=32):
    im = cv2.imread(filename)
    print(filename)
    # cv2.imshow('orig', im)
    res = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    # cv2.imshow('resize', res)
    # cv2.waitKey()
    im = (np.array(res))
    r = im[:,:, 0].flatten()
    g = im[:,:, 1].flatten()
    b = im[:,:, 2].flatten()

    out = np.array(list((label, )) + list(r) + list(g) + list(b), np.uint8)

    return out


def to_cifar(positive_dir, negative_dir, outfile, nbtest = 1000, w = 32, h = 32):
    data_dict = {}
    for file in os.listdir(positive_dir):
        if file.endswith(".png"):
            # print('Positive ' + file)
            data_dict[positive_dir + '/' + file] = 1

    for file in os.listdir(negative_dir):
        if file.endswith(".png"):
            # print('Negative ' + file)
            data_dict[negative_dir + '/' +  file] = 0

    keys = list(data_dict.keys())
    np.random.shuffle(keys)

    dataout = []

    # tests
    [dataout.append(to_bin(k, data_dict[k], w, h)) for k in keys[:nbtest]]
    dataout = np.concatenate(dataout, axis=0)
    # print(dataout)
    dataout =  np.array(dataout, np.uint8)
    dataout.tofile('test_' + outfile)

    dataout = []
    [dataout.append(to_bin(k, data_dict[k], w, h)) for k in keys[nbtest:]]
    dataout = np.concatenate(dataout, axis=0)
    # print(dataout)
    dataout =  np.array(dataout, np.uint8)
    dataout.tofile(outfile)

if __name__ == '__main__':
    # positive_dir negative_dir outputfile
    to_cifar(sys.argv[1], sys.argv[2], sys.argv[3])
