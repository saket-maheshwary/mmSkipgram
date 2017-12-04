from __future__ import print_function
import numpy as np
import ConfigParser
import utils
import os
import caffe
import scipy.io as sio
import skimage.io as skio
import glob
from copy import deepcopy
import time
import sys

def compute_cnn_features():
    """ Compute CNN features """

    settings = '../settings.ini'
    np.set_printoptions(threshold=np.nan)   # Print complete array
    print('Numpy:', np.__version__)
    config = ConfigParser.SafeConfigParser()
    config.read(settings)

    # Dataset
    dsetName = config.get('dataset', 'dset_name')
    dsetLoc = config.get('dataset', 'dset_loc')
    dsetLocPerm = config.get('dataset', 'dset_loc_perm')
    scratchLoc = config.get('system', 'scratch_loc')
    srcDir = config.get('system', 'srcDir')
    logDir = config.get('system', 'logDir')
    utils.mkdir_p(scratchLoc)
    utils.mkdir_p(dsetLoc)
    numCategories = int(config.get('dataset', 'num_categories'))

    # Settings related to experiment run
    exptName = config.get('expt', 'expt_name')
    randomSeed = int(config.get('expt', 'random_seed'))

    # Create output directory structure
    outDir = config.get('system', 'out_dir')
    outDirPerm = config.get('system', 'out_dir_perm')
    utils.mkdir_p(outDir)
    utils.mkdir_p(outDirPerm)
    outDir = outDir + os.sep + dsetName
    outDirPerm = outDirPerm + os.sep + dsetName
    utils.mkdir_p(outDir)
    utils.mkdir_p(outDirPerm)
    outDir = outDir + os.sep + exptName
    outDirPerm = outDirPerm + os.sep + exptName
    utils.mkdir_p(outDir)
    utils.mkdir_p(outDirPerm)

    # Other parameters
    FVDim = int(config.get('caffe', 'fv_dim'))
    FVLayerName = config.get('caffe', 'feature_layer')
    print(dsetName)
    print(dsetLoc)
    print(outDir)
    print(FVLayerName)

    # Serialization
    serializationDB = config.get('expt', 'serialization_db')
    serializationDB = outDir + os.sep + serializationDB

    # Communicating with matlab
    filelistDir = config.get('expt', 'filelist_dir')
    filelistDir = outDir + os.sep + filelistDir
    utils.mkdir_p(filelistDir)
    wordlistMat = config.get('expt', 'wordlist_mat')
    wordlistMat = filelistDir + os.sep + wordlistMat
    fvDir = config.get('expt', 'fv_dir')
    fvDir = outDir + os.sep + fvDir
    print(wordlistMat, os.path.exists(wordlistMat))
    utils.mkdir_p(fvDir)

    # Setup Caffe
    caffeMinibatchsize = int(config.get("caffe", "minibatchsize"))
    caffeMode = config.get("caffe", "mode")
    caffeGPUID = int(config.get("caffe", "gpu_id"))
    caffeModelName = config.get("caffe", "model_name")
    caffeModelFile = config.get("caffe", "model_file")
    caffePretrained = config.get("caffe", "pretrained")
    caffeImagesMean = config.get("caffe", "images_mean")
    caffeImageLabels = config.get("caffe", "image_labels")
    caffeImageSizeX = int(config.get("caffe", "image_size_x"))
    caffeImageSizeY = int(config.get("caffe", "image_size_y"))
    caffeImageCropSizeX = int(config.get("caffe",
					 "image_crop_size_x"))
    caffeImageCropSizeY = int(config.get("caffe",
					 "image_crop_size_y"))
    caffeFVDim = int(config.get("caffe", "fv_dim"))
    caffeFVLayerName = config.get('caffe', 'feature_layer')

    print('CAFFE mode       : ', caffeMode)
    print('CAFFE model name : ', caffeModelName)
    print('CAFFE model file : ', caffeModelFile)
    print('CAFFE pretrained : ', caffePretrained)

    if caffeMode.lower() == 'cpu'.lower():
	caffe.set_mode_cpu()
    else:
	caffe.set_mode_gpu()
	caffe.set_device(caffeGPUID)

    imagenetMean = np.load(caffeImagesMean).mean(1).mean(1)
    print(caffeModelFile)
    print(caffePretrained)
    print(imagenetMean)
    net = caffe.Classifier(caffeModelFile,
			   caffePretrained,
			   mean=imagenetMean,
			   channel_swap=(2, 1, 0),
			   raw_scale=255,
			   image_dims=(caffeImageCropSizeX,
				       caffeImageCropSizeY))

    modelLabels = np.loadtxt(caffeImageLabels, str, delimiter='\t')
    numImagesPerWord = 50
    wordlist = sio.loadmat(wordlistMat)
    wordsL = wordlist['words'].astype(str).tolist()
    words = [x.strip() for x in wordsL]
    numWords = len(words)
    idsL = wordlist['ids'].astype(str).tolist()
    ids = [x.strip() for x in idsL]
    print(words)

    for i in range(numWords):
       print('Processing %s i.e %s' % (words[i], ids[i]))
       npyPath = fvDir + os.sep + words[i] + '.npy'
       imgFV = np.zeros((numImagesPerWord, caffeFVDim), dtype=np.float32)
       fileGlob = '%s%s%s_*' % (dsetLoc, os.sep, ids[i])
       print(fileGlob)
       imgList = glob.glob(fileGlob)
       print(imgList)
       numImagesOfWord = len(imgList)
       for j in range(0, numImagesOfWord, caffeMinibatchsize):
           img_mb = []
           M = min(caffeMinibatchsize, numImagesPerWord-j)
           for k in range(M):
               im = deepcopy(caffe.io.load_image(imgList[j+k]))
               # print(im)
               # skio.imshow(im)
               # skio.show()
               img_mb.append(im)
           start_p = time.time()
           net.predict(img_mb)
           fv_mb = deepcopy(net.blobs[caffeFVLayerName].data)
           # print(fv_mb)
           imgFV[j:j+M] = np.squeeze(np.array(fv_mb,
                                     dtype=np.float32))[0:M,:]
           end_p = time.time()
           print('At %s image %d, time for batch of %d : %f sec'
                 % (words[i], i, M, end_p - start_p))
           sys.stdout.flush()
       # print(imgFV)
       np.save(npyPath, imgFV)
       print(npyPath, os.path.exists(npyPath))
