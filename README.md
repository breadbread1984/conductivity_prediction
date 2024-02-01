# Introduction

## install prerequisite packages

```shell
pip3 install -r requirements.txt
```

## generate dataset

### download polymer dataset

download the dataset from [bohrium](https://dataset-bohr-storage.dp.tech/lbg%2Fdataset%2Fzip%2Fdataset_tiefblue_bohr_14076_ai4scup-cns-5zkz_v101725.zip?Expires=1706793489&OSSAccessKeyId=LTAI5tGCcUT7wz9m1fq8cuLa&Signature=lJIRW1BiXYeKua7uGj293CT5WIo%3D) .

### generate tfrecord

```shell
mkdir dataset
python3 create_dataset.py --input_csv <path/to/mol_train.csv> --output_tfrecord dataset/trainset.tfrecord
python3 create_dataset.py --input_csv <path/to/mol_test.csv> --output_tfrecord dataset/testset.tfrecord
```

## train

```shell
python3 train.py --dataset dataset --ckpt <path/to/place/ckpt>
```

