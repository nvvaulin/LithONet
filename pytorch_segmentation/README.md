# Semantic Segmentation for facies in PyTorch

## Dataset
upload dataset from https://github.com/yalaudah/facies_classification_benchmark
```bash
# download the files: 
wget https://www.dropbox.com/s/p6cbgbocxwj04sw/data.zip
# check that the md5 checksum matches: 
openssl dgst -md5 data.zip # Make sure the result looks like this: MD5(data.zip)= bc5932279831a95c0b244fd765376d85, otherwise the downloaded data.zip is corrupted. 
# unzip the data:
unzip data.zip 
```
## Requirements
```bash
pip install -r requirements.txt
```

## Run train
```bash
pyhton3 train.py --config=config.yaml
```
