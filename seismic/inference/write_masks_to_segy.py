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
from torch.utils.data import DataLoader

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

from seismic.models.segmentation_benchmark.patch_deconvnet import patch_deconvnet
from seismic.models.baseline_unet import UnetResnet34
from seismic.config import config


def get_segy_info(filename):
    with segyio.open(filename, ignore_geometry=True) as f:
        # Get basic attributes
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000
        n_samples = f.samples.size
        twt = f.samples
    return n_traces, sample_rate, n_samples, twt


def predict_mask(image, model, dims=3, size=394, step=192, batch_size=8,
                 plot_image=False, dstdir=None, img_name='image1.png'):
    if image.ndim == 2:
        image = np.expand_dims(image, 2)
    if image.shape[-1] != dims:
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[-1] == 3:
            image = np.expand_dims(image[:, :, 0], 2)
    print(image.shape)

    # Cut large image into overlapping tiles
    tiler = ImageSlicer(image.shape, tile_size=(size, size), tile_step=(step, step), weight='pyramid')

    # HCW -> CHW. Optionally, do normalization here
    tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(image)]

    # Allocate a CUDA buffer for holding entire mask
    merger = CudaTileMerger(tiler.target_shape, 1, tiler.weight)

    # Run predictions for tiles and accumulate them
    for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)),
                                                batch_size=batch_size, pin_memory=True):
#         print(tiles_batch.shape)
        tiles_batch = tiles_batch.float().cuda()
        pred_batch = model(tiles_batch)
        pred_mask = pred_batch.max(dim=1)[1].float()

        merger.integrate_batch(pred_mask, coords_batch)

    # Normalize accumulated mask and convert back to numpy
    merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype(np.uint8)
    merged_mask = tiler.crop_to_orignal_size(merged_mask)

    if plot_image:
        assert dstdir is not None, 'dstdir should be passed'
        fig, ax = plt.subplots(ncols=2,figsize=(20,10))
        ax[0].imshow(image[:,:,0], cmap='gray')
        ax[1].imshow(merged_mask[:,:,0], alpha=0.3)
        fig.savefig(osp.join(dstdir, img_name), bbox_inches='tight', pad_inches=0)
        print(osp.join(dstdir, img_name))
    return merged_mask


def load_patch_model(weights_path, verbose=False):
    model = patch_deconvnet(n_classes=6)
    state_dict = torch.load(weights_path)['model']
    for param_name, param in model.named_parameters():
        state_weight = state_dict['module.' + param_name]
        if state_weight.shape == param.shape:
            if verbose:
                print(param_name, True)
            param = state_weight
        else:
            print(param_name, False)
    return model


def load_baseline_rosneft(weights_path):
    model = UnetResnet34(
        num_classes=8,
        pretrained=False
    )
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcpath', default=osp.join(config.data_dir, 'Parihaka3D', 'segys', 'Parihaka_PSTM_far_stack.sgy'),
                        help='path to input seismic .sgy file')
    parser.add_argument('--dstdir', default=osp.join(config.data_dir, 'outputs', 'segys'),
                        help='output directory')
    parser.add_argument('--dstfile', default='output_cude.sgy',
                        help='destination filename (only filename)')
    parser.add_argument('--model_arch', default='baseline_rosneft',
                        help='one of (baseline_patch_deconvnet, baseline_rosneft)')
    parser.add_argument('--weights_file', default=osp.join(config.models_dir, 'baselinev3', f'baseline_best.pth'),
                        help='path to file with model weights')
    parser.add_argument('--full_mode', action='store_true',
                        help='Default mode is test - we take only first iline image and first xline image from cube.'+\
                        ' This flag activates script which generates masks for full input cube (both iline and xline).')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.dstdir, exist_ok=True)

    if args.model_arch == 'baseline_patch_deconvnet':
        model = load_patch_model(args.weights_file)
        dims = 1
        size = 99
        step = 30
        batch_size = 32
    elif args.model_arch == 'baseline_rosneft':
        model = load_baseline_rosneft(args.weights_file)
        dims = 3
        size = 384
        step = 192
        batch_size = 8
    else:
        raise NotImplementedError('Only baseline_patch_deconvnet and baseline_rosneft archs are currently supported')

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

        mask = predict_mask(inline_img, model, dims=dims, size=size, step=step, batch_size=batch_size,
                            plot_image=True, dstdir=args.dstdir, img_name=f'inline{inline_idx}.png')

        xline_idx = segyfile_full.xlines[i]
        xline_img = segyfile_full.xline[xline_idx].T
        print('Shape of the xline_img', xline_img.shape)
        print('Starting inference of xline image...')

        mask = predict_mask(xline_img, model, dims=dims, size=size, step=step, batch_size=batch_size,
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
                    mask = predict_mask(iline_full_trn_cut, model_rosneft, dims=3, size=384, step=192)
                    dst.iline[iline_num] = mask
                print('Iline masks successfully generated!')

                print('Start generating xline masks...')
                for xline_num in tqdm.tqdm(spec.xlines):
                    image = src.xline[iline_num].T
                    mask = predict_mask(iline_full_trn_cut, model_rosneft, dims=3, size=384, step=192)
                    dst.xline[xline_num] = mask
                print('Xline masks successfully generated!')

        print('All done')

if __name__ == '__main__':
    main()
