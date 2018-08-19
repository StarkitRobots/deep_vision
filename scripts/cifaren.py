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
import shutil

def to_bin(filename, label, w, h, mode):
    im = cv2.imread(filename)
    #print(filename)
    res = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    if mode == "BGR":
        im = (np.array(res))
        r = im[:,:, 0].flatten()
        g = im[:,:, 1].flatten()
        b = im[:,:, 2].flatten()
        out = np.array(list((label, )) + list(r) + list(g) + list(b), np.uint8)
    elif mode == "Y":
        res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        im = (np.array(res))
        y = im[:,:].flatten()
        out = np.array(list((label, )) + list(y), np.uint8)
    return out


#def to_cifar(positive_dir, negative_dir, outfile, nbtest, w, h, mode):
#    data_dict = {}
#    for file in os.listdir(positive_dir):
#        if file.endswith(".png"):
#            # print('Positive ' + file)
#            data_dict[positive_dir + '/' + file] = 1
#
#    for file in os.listdir(negative_dir):
#        if file.endswith(".png"):
#            # print('Negative ' + file)
#            data_dict[negative_dir + '/' +  file] = 0
#
#    keys = list(data_dict.keys())
#    np.random.shuffle(keys)
#
#    dataout = []
#
#    # tests
#    [dataout.append(to_bin(k, data_dict[k], w, h, mode)) for k in keys[:nbtest]]
#    dataout = np.concatenate(dataout, axis=0)
#    # print(dataout)
#    dataout =  np.array(dataout, np.uint8)
#    dataout.tofile('test_' + outfile)
#
#    dataout = []
#    [dataout.append(to_bin(k, data_dict[k], w, h, mode)) for k in keys[nbtest:]]
#    dataout = np.concatenate(dataout, axis=0)
#    # print(dataout)
#    dataout =  np.array(dataout, np.uint8)
#    dataout.tofile(outfile)

def make_test_set(positive_dir,negative_dir):
    # Preparing the directories 
    positive_test_dir = positive_dir + 'test_set/'
    positive_learning_dir = positive_dir + 'learning_set/'
    negative_learning_dir = negative_dir + 'learning_set/'
    negative_test_dir = negative_dir + 'test_set/'

    for dir in [positive_test_dir,positive_learning_dir,negative_test_dir,negative_learning_dir]:
      if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(positive_test_dir)
    os.makedirs(positive_learning_dir)
    os.makedirs(negative_learning_dir)
    os.makedirs(negative_test_dir)

    # Splitting positive files in : learning/test
    pos_files = [file for file in os.listdir(positive_dir) if file.endswith("png")]
    np.random.shuffle(pos_files)

    nb_positive_tests = len(pos_files)/10;

    for file in pos_files[:nb_positive_tests]:
      shutil.copyfile(positive_dir + file,positive_test_dir + file)
    for file in pos_files[nb_positive_tests:]:
      shutil.copyfile(positive_dir + file,positive_learning_dir + file)

    # Adding noise in order to have more positive images
    os.system("python scripts/add_noise.py " + positive_test_dir)
    os.system("python scripts/add_noise.py " + positive_learning_dir)

    # Splitting negative files. We don't need to add noise
    neg_files = [file for file in os.listdir(negative_dir) if file.endswith("png")]
    np.random.shuffle(neg_files)
    nb_negative_tests = len(os.listdir(positive_test_dir + "/generated"))
    for file in neg_files[:nb_negative_tests]:
      shutil.copyfile(negative_dir + file,negative_test_dir + file)
    for file in neg_files[nb_negative_tests:]:
      shutil.copyfile(negative_dir + file,negative_learning_dir + file)

    return positive_test_dir + "generated/",positive_learning_dir + "generated/", negative_test_dir,negative_learning_dir

# In this version, the same number of pos and neg are used for tests
def to_cifar_balanced(positive_dir, negative_dir, outfile, w, h, mode):

    print "Preparing the learning and the test sets."
    pos_test_dir, pos_learning_dir, neg_test_dir, neg_learning_dir = make_test_set(positive_dir, negative_dir)

    print "Constructing the binaries."
    pos_test_files_path = [pos_test_dir + file for file in os.listdir(pos_test_dir) if file.endswith("png")]
    pos_learning_files_path = [pos_learning_dir + file for file in os.listdir(pos_learning_dir) if file.endswith("png")]
    neg_test_files_path = [neg_test_dir + file for file in os.listdir(neg_test_dir) if file.endswith("png")]
    neg_learning_files_path = [neg_learning_dir + file for file in os.listdir(neg_learning_dir) if file.endswith("png")]

    data_dict = {}
    for file in pos_test_files_path +  pos_learning_files_path:
        data_dict[file] = 1
    for file in neg_test_files_path + neg_learning_files_path:
        data_dict[file] = 0

    test_keys = pos_test_files_path + neg_test_files_path
    learning_keys = pos_learning_files_path + neg_learning_files_path
    np.random.shuffle(test_keys)
    np.random.shuffle(learning_keys)

    dataout = []

    # tests
    [dataout.append(to_bin(k, data_dict[k], w, h, mode)) for k in test_keys]
    dataout = np.concatenate(dataout, axis=0)
    # print(dataout)
    dataout =  np.array(dataout, np.uint8)
    dataout.tofile('test_' + outfile)

    dataout = []
    [dataout.append(to_bin(k, data_dict[k], w, h, mode)) for k in learning_keys]
    dataout = np.concatenate(dataout, axis=0)
    # print(dataout)
    dataout =  np.array(dataout, np.uint8)
    dataout.tofile(outfile)

if __name__ == '__main__':
    if (len(sys.argv) <= 1):
        print("Usage: <output_prefix> <opt: width (default 32)> <opt: height (default same as width)> <opt: mode (BGR or Y)>")
        exit()
    width = 32
    height = 32
    mode = "BGR"
    if (len(sys.argv) > 2):
        width = int(sys.argv[2])
        height = width
    if (len(sys.argv) > 3):
        height = int(sys.argv[3])
    if (len(sys.argv) > 4):
        mode = sys.argv[4]
    
    dir = glob.glob("Images/*")[0]
    positive_dir = dir + "/positive"
    negative_dir = dir + "/negative"
    to_cifar_balanced(positive_dir, negative_dir, sys.argv[1], width, height, mode)
