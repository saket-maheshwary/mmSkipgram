from __future__ import print_function
import time
import pdb
import logging
import sys
import numpy as np
import urllib2 as urllib
import os
from PIL import Image
import imghdr


class ImageNetDownloader(object):
    """Create a custom dataset from ImageNet"""

    def __init__(self):
        self.numClasses = 1000
        self.synsetListFile = \
            "/akashic/sourabh.d/installs/caffe/data/ilsvrc12/synsets.txt"
        self.numPerClassLimit = 10
        self.outDir = "/akashic/sourabh.d/datasets/ImageNetCustom"
        self.synsetQueryPrefix = \
            "http://www.image-net.org/api/text/" + \
            "imagenet.synset.geturls.getmapping?wnid="
        self.randomSeed = 1734

    def download(self):
        np.random.seed(seed=self.randomSeed)
        totalCnt = 0
        synsetList = np.genfromtxt(self.synsetListFile, dtype=str)
        print(synsetList)
        catCnt = 0
        for category in synsetList:
            t1s = time.time()
            c1s = time.clock()
            synsetQuery = self.synsetQueryPrefix + category
            print(synsetQuery)
            dwnldDir = self.outDir + os.sep + category
            os.mkdir(dwnldDir)
            filehandle = urllib.urlopen(synsetQuery)
            # print(filehandle.read())
            synsetImageList = \
                np.genfromtxt(filehandle, delimiter=' ', dtype=str)
            # print(synsetImageList)
            np.random.shuffle(synsetImageList)
            imgCnt = 0
            for imgID, imgURL in synsetImageList:
                # print(imgID)
                # print(imgURL)
                # print()
                try:
                    imgURLHandle = urllib.urlopen(imgURL)
                except urllib.HTTPError as e:
                    print('skipping due to HTTPError : %d' % e.code)
                    continue
                except urllib.URLError as e:
                    print('skipping due to URLError : %s' % e.reason)
                    continue
                except:
                    e = sys.exc_info()[0]
                    print('skipping due to exception: ', str(e))
                    continue
                fmt = os.path.splitext(imgURL)[-1]
                imgFileNameIn = dwnldDir + os.sep + imgID + fmt
                imgFileNameOut = dwnldDir + os.sep + imgID + ".jpg"
                print(imgFileNameIn)
                print(imgFileNameOut)
                try:
                    with open(imgFileNameIn, 'wb') as f:
                        f.write(imgURLHandle.read())
                except:
                    continue
                if imgFileNameIn != imgFileNameOut or \
                        imghdr.what(imgFileNameIn) != 'jpeg':
                    try:
                        im = Image.open(imgFileNameIn)
                        im.convert('RGB').save(imgFileNameOut, 'JPEG')
                        im.close()
                        if os.path.exists(imgFileNameIn):
                            os.remove(imgFileNameIn)
                    except:
                        print("skipping due to failure to open %s" %
                              imgFileNameIn)
                        if os.path.exists(imgFileNameIn):
                            os.remove(imgFileNameIn)
                        continue
                try:
                    imf = Image.open(imgFileNameOut)
                    imf.close()
                except:
                    print("skipping due improper jpeg %s" % imgFileNameOut)
                    if os.path.exists(imgFileNameOut):
                        os.remove(imgFileNameOut)
                    continue
                if imghdr.what(imgFileNameOut) != 'jpeg':
                    print("skipping due improper header : %s"
                          % imgFileNameOut)
                    if os.path.exists(imgFileNameOut):
                        os.remove(imgFileNameOut)
                    continue
                imgCnt = imgCnt + 1
                imgURLHandle.close()
                if imgCnt >= self.numPerClassLimit:
                    break
            filehandle.close()
            totalCnt = totalCnt + imgCnt
            t1e = time.time()
            c1e = time.clock()
            print('%d:: %s : %d images : wall time %f s : proc time %f s '
                  % (catCnt, category, imgCnt,
                     t1e-t1s, c1e-c1s))
            catCnt = catCnt + 1
        print('Total: %d' % totalCnt)


if __name__ == '__main__':
    start_time = time.time()
    start_ctime = time.clock()
    try:
        dldr = ImageNetDownloader()
        dldr.download()
        end_time = time.time()
        end_ctime = time.clock()
        print('Success: wall time:  %f sec, processor time: %f sec'
              % (end_time-start_time, end_ctime-start_ctime))
    except:
        end_time = time.time()
        end_ctime = time.clock()
        print('Failure: wall time: %f sec, processor time: %f sec'
              % (end_time-start_time, end_ctime-start_ctime))
        logging.exception(sys.exc_info()[2])
        pdb.post_mortem(sys.exc_info()[2])
