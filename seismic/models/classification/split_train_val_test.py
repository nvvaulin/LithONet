import os
import os.path as osp
import pandas as pd

from sklearn.model_selection import train_test_split

from seismic.config import config

def make_img_info(data_dir):
    all_dict = {'files': [], 'target': []}
    for root, dirs, files in os.walk(data_dir):
        if len(files) > 0:
            cut_root = root[len(data_dir)+1:]
            files = [osp.join(cut_root, file) for file in files if file.endswith('.mat')]
            all_dict['files'].extend(files)
            all_dict['target'].extend([cut_root] * len(files))
    return pd.DataFrame(all_dict)

if __name__ == '__main__':
    for subdir in ['LANDMASS1', 'LANDMASS2']:
        data_dir = osp.join(config.data_dir, 'LANDMASS', subdir)


        img_info = make_img_info(data_dir)

        train, test = train_test_split(img_info, test_size=0.1, random_state=24, stratify=img_info['target'])
        train, val = train_test_split(train, test_size=0.2, random_state=24, stratify=train['target'])

        print(subdir, train.shape, val.shape, test.shape)
        train.to_csv(osp.join(data_dir, 'train.csv'), index=False)
        val.to_csv(osp.join(data_dir, 'val.csv'), index=False)
        test.to_csv(osp.join(data_dir, 'test.csv'), index=False)
