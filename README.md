# Aramco Technathon (December 2019)

## Challenge 1: AI Challenge: Automated Seismic Interpretation

### Run instructions

#### 1. Clone repository and checkout to the correct branch
```git clone https://github.com/nvvaulin/LithONet.git
cd LithONet
git fetch origin
git checkout features/rewrite_all
```

#### 2. Create virtual environment in your favorite Python package manager and run:
```pip install requirements.txt
pip install -e .
```

#### 3. Download models  

Download link - . Put `models/` folder to the root directory of this repo.

#### 4. Data preparation

Put `Parihaka3D` folder (or some .sgy file) to `data/` folder.

Now you can run scripts. Run inference script:
```python seismic/inference/write_masks_to_segy.py
```
OR
```python seismic/inference/write_masks_to_segy.py --srcpath <path_to_your_.sgy_file>
```

To gather predictions for entire cube, run
```# This operation will take ~4 hours
python seismic/inference/write_masks_to_segy.py --srcpath <path_to_your_.sgy_file> --full_mode
```


### Help

Parameters for our inference script:
```usage: write_masks_to_segy.py [-h] [--srcpath SRCPATH] [--dstdir DSTDIR]
                              [--dstfile DSTFILE] [--model_arch MODEL_ARCH]
                              [--weights_file WEIGHTS_FILE] [--full_mode]

optional arguments:
  -h, --help            show this help message and exit
  --srcpath SRCPATH     path to input seismic .sgy file
  --dstdir DSTDIR       output directory
  --dstfile DSTFILE     destination filename (only filename)
  --model_arch MODEL_ARCH
                        one of (baseline_patch_deconvnet, baseline_rosneft)
  --weights_file WEIGHTS_FILE
                        path to file with model weights
  --full_mode           Default mode is test - we take only first iline image
                        and first xline image from cube. This flag activates
                        script which generates masks for full input cube (both
                        iline and xline).
```
