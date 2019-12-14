import os.path as osp
import os
import torch
from torchvision import transforms

class Config():
    basedir = osp.abspath(osp.dirname(__file__))
    CURRENT_PATH = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(CURRENT_PATH, '..')

    models_dir = osp.join(root_dir, 'models')
    data_dir = osp.join(root_dir, 'data')
    runs_dir = osp.join(root_dir, 'runs')
    submissions_dir = osp.join(root_dir, 'submissions')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LandmassConfig():
    class_name_to_id = {'chaotic': 0, 'fault': 1, 'horizon': 2, 'salt dome': 3}
    id_to_class_name = dict((v, k) for k, v in class_name_to_id.items())

    basic_transforms = [transforms.ToTensor()]

config = Config()
landmass_config = LandmassConfig()
