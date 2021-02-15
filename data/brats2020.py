import os
import torch

import SimpleITK as sitk
import numpy as np
np.random.seed = 0

from .data_provider_base import DataProviderBase
from .base import DataGeneratorBase
from metrics import BRATSMetric
from parser import brain_tumor_argparse

parser = brain_tumor_argparse()
parse_args = parser.parse_args()

from dotenv import load_dotenv

label_base = 'seg'
load_dotenv('./.env')

BRATS2020_DIR = os.environ.get('BRATS2020_DIR')
BRATS2020_DIR_TEST = os.environ.get('BRATS2020_DIR_VALIDATION')
BRATS2020_TRAIN = BRATS2020_DIR
BRATS2020_VALIDATION = BRATS2020_DIR_TEST

class Brats2020DataProvider(DataProviderBase):

    def __init__(self, args):
        self._metric = BRATSMetric

        self.data_dirs = self._get_dirs(args)
        
        self.data_test_dirs = self._get_test_dirs(args)

        self.modal_bases = self._get_modal_bases(args)

        self.all_ids = self._get_all_ids()

        # self.train_ids = self.all_ids
        # self.test_ids = self._get_test_all_ids()
        self.train_ids = self.all_ids[: -len(self.all_ids) // 10]
        self.test_ids = self.all_ids[-len(self.all_ids) // 10:]

        print(f'training on {len(self.train_ids)} samples, '
              f'validating on {len(self.test_ids)} samples')

    def _get_all_ids(self):
        all_ids = []
        for data_dir in self.data_dirs:
            folder_names = os.listdir(data_dir)
            folder_dirs = [os.path.join(data_dir, foldername) for foldername in folder_names]
            all_ids.extend(folder_dirs)
        return all_ids
    
    def _get_test_all_ids(self):
        all_ids = []
        for data_dir in self.data_test_dirs:
            folder_names = os.listdir(data_dir)
            folder_dirs = [os.path.join(data_dir, foldername) for foldername in folder_names]
            all_ids.extend(folder_dirs)
        return all_ids

    def _get_raw_data_generator(self, data_ids, **kwargs):
        return Brats2020DataGenerator(data_ids, self.data_format, self.modal_bases, **kwargs)

    def _get_dirs(self, args):
        data_dirs = []
        data_dirs = [BRATS2020_TRAIN]
        return data_dirs
    
    def _get_test_dirs(self, args):
        data_dirs = []
        data_dirs = [BRATS2020_VALIDATION]
        return data_dirs

    def _get_modal_bases(self, args):
        modal_bases = []
        if 'flair' in args:
            modal_bases.append('_flair.')
        if 't1' in args:
            modal_bases.append('_t1.')
        if 't1ce' in args:
            modal_bases.append('_t1ce.')
        if 't2' in args:
            modal_bases.append('_t2.')
        if not modal_bases:
            modal_bases = ['_flair.', '_t1.', '_t1ce.', '_t2.']
        return modal_bases

    @property
    def data_format(self):
        return {
            "channels": len(self.modal_bases),
            "depth": 155,
            "height": 240,
            "width": 240,
            "class_num": 5,
        }


class Brats2020DataGenerator(DataGeneratorBase):

    def __init__(self, data_ids, data_format, modal_bases, random=True, **kwargs):
        super().__init__(data_ids, data_format, random)
        self.modal_bases = modal_bases

    def _get_image_and_label(self, data_id):
        image = [self._get_image_from_folder(data_id, base) for base in self.modal_bases]
        image = np.asarray(image)
        label = self._get_image_from_folder(data_id, label_base)
        return image, label

    @staticmethod
    def _get_image_from_folder(folder_dir, match_string):
        modal_folder = []
        for f in os.listdir(folder_dir):
          if isinstance(f, str):
            if match_string in f:
              modal_folder.append(f)
          else:
            if match_string in f.decode("utf-8"):
              modal_folder.append(f)

        assert(len(modal_folder) == 1)

        modal_folder_dir = os.path.join(folder_dir, modal_folder[0] if isinstance(modal_folder[0], str) else modal_folder[0].decode("utf-8"))

        data_filename = modal_folder_dir

        image = sitk.ReadImage(data_filename)
        image_array = sitk.GetArrayFromImage(image)
        return image_array

    def _get_data(self, data_ids):
        batch_volume = np.empty((
            len(data_ids),
            self.data_format['channels'],
            self.data_format['depth'],
            self.data_format['height'],
            self.data_format['width'],
        ))
        batch_label = np.empty((
            len(data_ids),
            self.data_format['depth'],
            self.data_format['height'],
            self.data_format['width'],
        ), dtype=np.uint8)

        for idx, data_id in enumerate(data_ids):
            batch_volume[idx], batch_label[idx] = self._get_image_and_label(data_id)
        return {'volume': batch_volume, 'label': batch_label, 'data_ids': data_ids}
