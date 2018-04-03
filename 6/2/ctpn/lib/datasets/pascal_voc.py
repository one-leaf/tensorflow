# -*- coding:utf-8 -*-
import os
import numpy as np
import scipy.sparse
try:
    import cPickle as pickle
except:
    import pickle
#import pickle
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
from .imdb import imdb
from . import ds_utils
from ..fast_rcnn.config import cfg
'''
<annotation>
    <folder>VOC2007</folder>
    <filename>009961.jpg</filename>
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>334575803</flickrid>
    </source>
    <owner>
        <flickrid>dictioncanary</flickrid>
        <name>Lucy</name>
    </owner>
    <size><!--image shape-->
        <width>500</width>
        <height>374</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented><!--是否有分割label-->
    <object>
        <name>dog</name> <!--类别-->
        <pose>Unspecified</pose><!--物体的姿态-->
        <truncated>0</truncated><!--物体是否被部分遮挡（>15%）-->
        <difficult>0</difficult><!--是否为难以辨识的物体， 主要指要结体背景才能判断出类别的物体。虽有标注， 但一般忽略这类物体-->
        <bndbox><!--bounding box-->
            <xmin>69</xmin>
            <ymin>4</ymin>
            <xmax>392</xmax>
            <ymax>345</ymax>
        </bndbox>
    </object>
</annotation>

分割的label由两部分组成： 
* class segmentation: 标注出每一个像素的类别 
* object segmentation: 标注出每一个像素属于哪一个物体 
'''
class pascal_voc(imdb):
    # 初始化 pascal_voc 
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        # ctpn 的分类，背景和文字【0，1】
        self._classes = ('__background__', # always index 0
                         'text')
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        #self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,   # 是否使用难以识别的样本
                       'matlab_eval' : False,   # 是否采用 matlab 来计算
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    # 返回第 i 张图片的相对路径
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    # 按文件名（不含扩展名），返回图片的相对路径
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # 加载所有图片文件名（不含扩展名）的清单
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    # 获得数据集的存放路径
    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    # 获得真实样本的数据集
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    # 加载 rpn 产生的 roi
    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    # 读取 rpn 文件
    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in range(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    # 读取 gt 数据
    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            '''
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            '''
            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

     def _get_voc_results_file_template(self):
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    # 将 VOC 的检测结果写入文件
    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))


if __name__ == '__main__':
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
