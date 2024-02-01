# Introduction

## install prerequisite packages

```shell
pip3 install -r requirements.txt
```

if version of python is over 3.9 (included), please manually update from collection to collection.abc for the following files

- /home/breadbread1984/envs/tf2/lib/python3.11/site-packages/mordred/tests/test_import_all_descriptors.py:1
- /home/breadbread1984/envs/tf2/lib/python3.11/site-packages/mordred/tests/test_result_type.py:1

modred has a bug, please manual update /home/breadbread1984/envs/tf2/lib/python3.11/site-packages/mordred/DetourMatrix.py:121 as the following code

```python
        for bcc in (self.G.subgraph(c) for c in networkx.biconnected_components(self.G)):
```

verify your environment by executing the following command

```shell
python3 -m mordred.tests
```

if no errors occur, you can safely generated mordred descriptors.

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

