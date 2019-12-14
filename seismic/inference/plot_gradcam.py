import argparse
import re
import numpy as np
import pandas as pd
import segyio
import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

from seismic.models.classification.transform import scale_img
from seismic.models.classification.gradCam import show_gradcam
from seismic.config import config, landmass_config


def get_segy_info(filename):
    with segyio.open(filename, ignore_geometry=True) as f:
        # Get basic attributes
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000
        n_samples = f.samples.size
        twt = f.samples
    return n_traces, sample_rate, n_samples, twt


def predict_gradcam_mask(image, model, dims=3, size=(150, 300), step=(150, 300),
                         batch_size=8, grad_thr=0.6, weight_type='mean',
                         plot_image=False, dstdir=None, img_name='image1.png'):
    image = scale_img(image).astype(np.float32)

    if image.ndim == 2:
        image = np.expand_dims(image, 2)
    if image.shape[-1] != dims:
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[-1] == 3:
            image = np.expand_dims(image[:, :, 0], 2)

    image = (image - np.min(image))/ (0.5 * np.ptp(image)) - 1

    # Cut large image into overlapping tiles
    tiler = ImageSlicer(image.shape, tile_size=size, tile_step=step, weight=weight_type)

    # HCW -> CHW. Optionally, do normalization here
    tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(image)]

    # Allocate a CUDA buffer for holding entire mask
    merger = CudaTileMerger(tiler.target_shape, 1, tiler.weight)

    # Run predictions for tiles and accumulate them
    for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)),
                                                batch_size=batch_size, pin_memory=True):
        tiles_batch = tiles_batch.float().cuda()
        with torch.no_grad():
            pred_batch = torch.max(F.softmax(model(tiles_batch), dim=1),
                                   dim=1)[1].detach().cpu().numpy()
        image_needed_classes = pred_batch == 1
        masks = []
        for tile_idx, has_target in enumerate(image_needed_classes):
            if has_target:
                tile = tiles_batch[tile_idx].unsqueeze(0)
                heatmap, mask = show_gradcam(tile, model)
                mask = mask[:,:,0]
                mask[mask < grad_thr] = 0
                masks.append(torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
            else:
                masks.append(torch.zeros_like(tiles_batch[tile_idx, 0]).unsqueeze(0).unsqueeze(0))
        masks = torch.cat([mask.cuda() for mask in masks], dim=0) * 1000

        merger.integrate_batch(masks, coords_batch)

    # Normalize accumulated mask and convert back to numpy
    merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype(np.uint8)
    merged_mask = tiler.crop_to_orignal_size(merged_mask) / 1000

    if plot_image:
        assert dstdir is not None, 'dstdir should be passed'
        fig, ax = plt.subplots(ncols=2,figsize=(20,10))
        ax[0].imshow(image[:,:,0], cmap='gray')
        ax[1].imshow(merged_mask[:,:,0], alpha=0.3)
        fig.savefig(osp.join(dstdir, img_name), bbox_inches='tight', pad_inches=0)
        print(osp.join(dstdir, img_name))
    return merged_mask


def load_clf_model(weights_path):
    n_classes = len(landmass_config.class_name_to_id)
    model = resnet18(pretrained=False, num_classes=n_classes)
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcpath', default=osp.join(config.data_dir, 'Parihaka3D', 'segys', 'Parihaka_PSTM_far_stack.sgy'),
                        help='path to input seismic .sgy file')
    parser.add_argument('--dstdir', default=osp.join(config.data_dir, 'outputs', 'gradcam_mask'),
                        help='output directory')
    parser.add_argument('--dstfile', default='output_cude.sgy',
                        help='destination filename (only filename)')
    parser.add_argument('--weights_file', default=osp.join(config.models_dir, 'resnet18',
                                                           '2019-12-13_00-13-22', 'model_best.pkl'),
                        help='path to file with model weights')
    parser.add_argument('--full_mode', action='store_true',
                        help='Default mode is test - we take only first iline image and first xline image from cube.'+\
                        ' This flag activates script which generates masks for full input cube (both iline and xline).')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.dstdir, exist_ok=True)

    model = load_clf_model(args.weights_file)
    dims = 3
    size = (150, 300)
    step = (150, 300)
    batch_size = 8

    model.eval()
    model.to(config.device)

    n_traces, sample_rate, n_samples, twt = get_segy_info(args.srcpath)
    print('Full_cube')
    print('N Traces: {}, N Samples: {}, Sample rate: {}ms, Trace length: {}'.format(n_traces,
                                                                                    n_samples,
                                                                                    sample_rate,
                                                                                    max(twt)))
    if not args.full_mode:
        segyfile_full = segyio.open(args.srcpath, 'r')
        segyfile_full.mmap()

        print('Full cube')
        print('  Crosslines: ', segyfile_full.xlines[0], ':', segyfile_full.xlines[-1], ' | lngth:',len(segyfile_full.xlines))
        print('  Inlines:    ', segyfile_full.ilines[0], ':', segyfile_full.ilines[-1], ' | lngth:',len(segyfile_full.ilines))

        i = 0
        inline_idx = segyfile_full.ilines[i]
        inline_img = segyfile_full.iline[inline_idx].T
        print('Shape of the iline_img', inline_img.shape)
        print('Starting inference for iline image...')

        mask = predict_gradcam_mask(inline_img, model, dims=dims, size=size, step=step, batch_size=batch_size,
                            plot_image=True, dstdir=args.dstdir, img_name=f'inline{inline_idx}.png')

        xline_idx = segyfile_full.xlines[i]
        xline_img = segyfile_full.xline[xline_idx].T
        print('Shape of the xline_img', xline_img.shape)
        print('Starting inference of xline image...')

        mask = predict_gradcam_mask(xline_img, model, dims=dims, size=size, step=step, batch_size=batch_size,
                            plot_image=True, dstdir=args.dstdir, img_name=f'xline{xline_idx}.png')
        print(f'Plots are saved in {args.dstdir}')

    else:
        dstpath = osp.join()
        with segyio.open(srcpath) as src:
            spec = segyio.spec()
            spec.sorting = src.sorting
            spec.format = src.format
            spec.samples = src.samples
            spec.ilines = src.ilines
            spec.xlines = src.xlines
            with segyio.create(dstpath, spec) as dst:
                dst.text[0] = src.text[0]

                print('Start generating iline masks...')
                for iline_num in tqdm.tqdm(spec.ilines):
                    image = src.iline[iline_num].T
                    mask = predict_gradcam_mask(iline_full_trn_cut, model_rosneft, dims=dims, size=size, step=step)
                    dst.iline[iline_num] = mask
                print('Iline masks successfully generated!')

                print('Start generating xline masks...')
                for xline_num in tqdm.tqdm(spec.xlines):
                    image = src.xline[iline_num].T
                    mask = predict_gradcam_mask(iline_full_trn_cut, model_rosneft, dims=dims, size=size, step=step)
                    dst.xline[xline_num] = mask
                print('Xline masks successfully generated!')

        print('All done')

if __name__ == '__main__':
    main()
