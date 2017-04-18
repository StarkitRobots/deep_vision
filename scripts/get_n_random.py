#!/usr/bin/python
# -*- coding: utf-8 -*-

#
#  File Name	: get_n_random.py
#  Author	: Steve NGUYEN
#  Contact      : steve.nguyen.000@gmail.com
#  Created	: mercredi, avril 12 2017
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

import numpy as np
import glob
import os
import shutil

outname = '/sample/'
files = glob.glob(sys.argv[1] + '*.png')

if not os.path.exists(sys.argv[1] + outname):
    os.makedirs(sys.argv[1] + outname)


np.random.shuffle(files)
i = 0
for i in range(int(sys.argv[2])):
    f = files[i]
    # print os.path.dirname(f), os.path.basename(f)
    print os.path.dirname(f) + outname + os.path.basename(f)

    shutil.copy(
        os.path.dirname(f) + '/' + os.path.basename(f), os.path.dirname(f) + outname + os.path.basename(f))
