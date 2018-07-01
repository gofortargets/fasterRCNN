
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.utils.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid
# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

class thilini(imdb):
    def __init__(self, data_type):
        print ('Init class for', data_type)
        imdb.__init__(self, data_type)

        # # COCO specific config options
        # self.config = {'use_salt': True,
        #                'cleanup': True}

        self._data_path = osp.join(cfg.DATA_DIR, 'thilini')
        annotation_file_path = osp.join(self._data_path, data_type + '.json')
        # if not annotation_file_path == None:

        assert annotation_file_path != None
        # if self._dataset == None:
        print('loading annotations ', annotation_file_path, 'into memory...')
        tic = time.time()
        dataset = json.load(open(annotation_file_path, 'r'))
        print('Done (t=%0.2fs)'%(time.time()- tic))

        self._dataset = dataset
        self._image_index = self._load_image_set_index()
        self.getClasses()
        print('Total vocab =', len(self.classes), self.num_classes)
        print('Top vocab =', self.classes[:10])

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._roidb_handler = self.gt_roidb
        print ('Done Init class')

    # imgs = {}
    def getClasses(self):
        print('Geting classes...')

        cache_file = os.path.join(self.cache_path, self.name + '_gt_classes.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self._classes = pickle.load(fid)
                print ('len of loaded classes =', len(self._classes))
            print('{} gt classes loaded from {}'.format(self.name, cache_file))
            return

        self._classes = set()
        for imageID in self._dataset.keys():
           for region in self._dataset[imageID]['region_list']:
              self._classes.add(region['verb'])

        self._classes = list(self._classes)
        with open(cache_file, 'wb') as fid:
            pickle.dump(self._classes, fid, pickle.HIGHEST_PROTOCOL)

    def gt_roidb(self):
        print ('Getting roidb from raw dataset ...')
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        tic = time.time()
        gt_roidb = []
        for id, index in enumerate(self.image_index):
           gt_roidb.append(self._load_annotation(index))
           if id % 1000 == 0:
               print ('Current at', id)

        print('Done (t=%0.2fs)'%(time.time()- tic))

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print ('Done')
        return gt_roidb

    def _load_annotation(self, index):
        # print ('self.num_classes =', self.num_classes)
        im_ann = self._dataset[index]
        width = int(im_ann['width'])
        height = int(im_ann['height'])
        bboxs_info = im_ann['region_list']
        num_boxes = len(bboxs_info)

        # boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        # gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_boxes), dtype=np.float32)
        boxes = []
        gt_classes = []

        for ix, box in enumerate(bboxs_info):
            bbox = box['region_bbox']
            x1 = np.max((0, int(bbox['x'])))
            y1 = np.max((0, int(bbox['y'])))
            x2 = np.min((width - 1, x1 + np.max((0, int(bbox['w']) - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, int(bbox['h']) - 1))))
            boxes.append([x1, y1, x2, y2])
            cls = self._class_to_ind[box['verb']]
            assert cls < self.num_classes
            gt_classes.append(cls)
            overlaps[ix][cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        gt_classes = np.array(gt_classes)
        boxes = np.array(boxes)

        #sanitize bboxes
        # if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
        #     obj['clean_bbox'] = [x1, y1, x2, y2]
        #     valid_objs.append(obj)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'dataset', index)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def _load_image_set_index(self):
        # image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
        #                               self._image_set + '.txt')
        # assert os.path.exists(image_set_file), \
        #     'Path does not exist: {}'.format(image_set_file)
        # with open(image_set_file) as f:
        #     image_index = [x.strip() for x in f.readlines()]
        # return image_index
        return list(self._dataset.keys())
        # pass
