#!/usr/bin/python
# -*- coding: utf-8 -*-

#
#  File Name	: 'get_data.py'
#  Author	: Steve NGUYEN
#  Contact      : steve.nguyen.000@gmail.com
#  Created	: vendredi, octobre 28 2016
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

from lxml import etree
import sys
import cv2
import numpy as np

# tree = etree.parse(sys.argv[1])
# basedir = sys.argv[2]

tags_files = ['leipz1/tags.xml',  'leipz3/tags.xml',
              'leipz4/tags.xml', 'leipz5/tags.xml', 'leipz_from_side/tags.xml']
directories = ['leipz1/',  'leipz3/',
               'leipz4/', 'leipz5/', 'leipz_from_side/']


bigmama = 'big_mama.xml'

# camera_matrix = np.asarray(
#     cv2.cv.Load(bigmama, cv2.cv.CreateMemStorage(), 'camera_matrix'))
# dist_coeffs = np.asarray(
# cv2.cv.Load(bigmama, cv2.cv.CreateMemStorage(),
# 'distortion_coefficients'))

map1 = np.asarray(
    cv2.cv.Load(bigmama, cv2.cv.CreateMemStorage(), 'map1'))
map2 = np.asarray(
    cv2.cv.Load(bigmama, cv2.cv.CreateMemStorage(), 'map2'))

nbball = 0
nbnoball = 0

MODE = 0  # 0 positive or 1 negative data

for tags, basedir in zip(tags_files, directories):

    print tags, basedir
    tree = etree.parse(tags)
    roottag = tree.getroot().tag
    nodes = []
    dependencies = {}
    classname = {}
    pipename = {}
    colordep = {}
    print roottag
    balls = []
    noballs = []
    for pipe in tree.xpath('/' + roottag):

        for fil in pipe:
            # print
            # print("NEW FILTER")
            name = ''
            pipe = ''
            noBall = False
            for child in fil:
                radius = 0
                row = 0
                col = 0
                ball = []
                # print ("%s %s" % (child.tag, child.text))

                if child.tag == 'name':
                    # ball.append(child.text)  # filename
                    # print child.text
                    name = child.text

                elif child.tag == 'tag':
                    for cc in child:
                        # print cc.tag
                        if cc.tag == 'Ball':
                            for ccc in cc:
                                # print ccc.tag
                                if ccc.tag == 'noBall':
                                    noBall = True
                                    print "noBall", name
                                    noballs.append(name)
                                    # get ball_negative data

                                    break
                                else:

                                    if ccc.tag == 'radius':
                                        radius = ccc.text
                                        print "radius ", radius

                                    elif ccc.tag == 'center':
                                        for cccc in ccc:
                                            # print cccc.tag
                                            if cccc.tag == 'row':
                                                row = cccc.text
                                            else:
                                                col = cccc.text

                ball.append(name)
                if not noBall:

                    ball.append(row)
                    ball.append(col)
                    ball.append(radius)

            balls.append(ball)

        print balls

    # images
    window_size = 50

    if MODE == 0:

        keepball = 0

        for im in balls:
            print im
            if len(im) == 4:
                image = cv2.imread(basedir + '/' + im[0])
                row = int(im[1])
                col = int(im[2])
                rad = float(im[3])

                if (row - window_size) < 0 or (row + window_size) > image.shape[0] or (col - window_size) < 0 or (col + window_size) > image.shape[1]:
                    print 'ignore'
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)
                # cv2.imshow("test", image)
                dst = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

                # np.random.randint(1, 15)

                data = dst[row - window_size: row + window_size,
                           dst.shape[1] - col - window_size:dst.shape[1] - col + window_size]
                nbball += 1
                if (nbball % 10) == 0:  # reserve some data for tests
                    cv2.imwrite('keep_%06d.png' % (nbball), data)
                    continue

                cv2.imwrite('%06d.png' % (nbball), data)

                # cv2.imshow('test box', data)
                # cv2.waitKey()

                # cv2.rectangle(dst, (dst.shape[1] - col - window_size, row - window_size), (
                # dst.shape[1] - col + window_size, row + window_size), (0, 0,
                # 255), 2)

                for i in range(50):
                    noisex = np.random.randint(- 40, 40)
                    noisey = np.random.randint(- 40, 40)
                    angle = np.random.randint(- 15, 15)

                    # M = cv2.getRotationMatrix2D(
                    #     (data.shape[0] / 2, data.shape[1] / 2), angle, 1)
                    # data = cv2.warpAffine(data, M, (data.shape[0],
                    # data.shape[1]))

                    M = cv2.getRotationMatrix2D(
                        (dst.shape[1] - col, row), angle, 1)
                    dd = cv2.warpAffine(dst, M, (dst.shape[1], dst.shape[0]))

                    dd = dd[
                        row - window_size + noisey: row + window_size + noisey,
                        dd.shape[1] - col - window_size + noisex:dd.shape[1] - col + window_size + noisex]

                    # cv2.imshow("rotate", data)
                    # cv2.waitKey()

                    # data = dst[row - window_size + noisey: row + window_size + noisey,
                    # dst.shape[1] - col - window_size + noisex:dst.shape[1] - col +
                    # window_size + noisex]

                    # also add noise on brightness
                    hsv = cv2.cvtColor(dd, cv2.COLOR_BGR2HSV)

                    h, s, v = cv2.split(hsv)
                    # print np.min(v), np.max(v)
                    noise = np.random.randint(- 40, 10)
                    # noise = np.random.rand()
                    # print v, v.shape
                    # v = np.floor(v / noise)
                    # v = v.astype(int)
                    # v = v / 2
                    v += noise
                    v = np.clip(v, 0, 255)
                    # print v, v.shape, np.min(v), np.max(v), noise
                    hsv = cv2.merge((h, s, v))

                    # hsv[:, :, 2] += np.random.randint(- 100, 0)
                    imgout = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                    cv2.imwrite('%06d.png' % (nbball), imgout)
                    nbball += 1
                # cv2.rectangle(dst, (dst.shape[1] - col - int(rad), row - int(rad)), (
                #     dst.shape[1] - col + int(rad), row + int(rad)), (0, 0, 255), 2)
                # cv2.imshow("undistort", dst)

                # cv2.waitKey()
    else:
        # dump negative data

        print "DATA negative"
        # print noballs
        for im in noballs:
            print im

            image = cv2.imread(basedir + '/' + im)

            image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)
            # cv2.imshow("test", image)
            dst = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

            # np.random.randint(1, 15)

            for i in range(300):

                col = np.random.randint(
                    window_size, dst.shape[1] - window_size)
                row = np.random.randint(
                    window_size, dst.shape[0] - window_size)
                # print 'rand', col, row, dst.shape
                data = dst[row - window_size: row + window_size,
                           dst.shape[1] - col - window_size:dst.shape[1] - col + window_size]

                cv2.imwrite('%06d.png' % (nbnoball), data)

                # cv2.imshow('negative', data)
                nbnoball += 1
                # cv2.waitKey()
