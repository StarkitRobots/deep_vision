#!/usr/bin/python
# -*- coding: utf-8 -*-

#
#  File Name	: 'test_images.py'
#  Author	: Steve NGUYEN
#  Contact      : steve.nguyen.000@gmail.com
#  Created	: vendredi, mars 10 2017
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
from subprocess import call
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':
    result = {}
    basebin = sys.argv[1]
    basetest = os.path.dirname(sys.argv[2])

    files = [basetest + '/' + f for f in listdir(
        basetest) if isfile(join(basetest, f))]

    print "runnin: ", basebin

    for img in files:
        print basebin,  img
        res = call([basebin, img])
        print "Output class: ", res
        result[img] = res

    # print result

    nb_pos = 0
    nb_neg = 0
    tot = 0
    posimg = []

    negimg = []

    for f, c in result.iteritems():
        if c == 1:
            nb_pos += 1
            posimg.append(f)

        else:
            nb_neg += 1
            negimg.append(f)

    print "________"
    print "POS: ",  nb_pos
    print "NEG: ",  nb_neg
    # print negimg

    # for neg in negimg:
    #     shutil.copy(neg, basetest + "/neg/" + os.path.basename(neg))

    # for pos in posimg:
    #     shutil.copy(pos, basetest + "/pos/" + os.path.basename(pos))
