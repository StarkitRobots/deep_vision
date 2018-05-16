#!/usr/bin/python
# -*- coding: utf-8 -*-

#
#  File Name	: 'get_from_json.py'
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

import json
import os
import sys
import shutil

if __name__ == '__main__':
    base = os.path.dirname(sys.argv[1])
    with open(sys.argv[1], 'r') as datafile:
        balls = json.load(datafile)

        for b in balls:
            shutil.move(
                base + '/' + b, base + "/positive/" + b)
