#!/usr/bin/python
# -*- coding: utf-8 -*-

#
#  File Name	: 'dump_patches.py'
#  Author	: Steve NGUYEN
#  Contact      : steve.nguyen.000@gmail.com
#  Created	: DDDD
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
from subprocess import call
import fnmatch


arch = "../bin/models/test_exp.json"
weights = "../bin/models/test_exp_weights.bin"

if __name__ == '__main__':
    result = {}
    basebin = sys.argv[1]
    basetest = sys.argv[2]
    out_dir = sys.argv[3]

    matches = []
    for root, dirnames, filenames in os.walk(basetest):
        for filename in fnmatch.filter(filenames, '*.png'):
            matches.append(os.path.join(root, filename))

    # print matches

    # files = [basetest + '/' + f for f in listdir(
    #     basetest) if isfile(join(basetest, f))]
    print "running: ", basebin

    for img in matches:
        print img
        res = call([basebin, arch, weights, img, out_dir])
