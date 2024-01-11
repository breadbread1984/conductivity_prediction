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

## generate dataset

### download polymer dataset

download the dataset from [PI1M](https://github.com/RUIMINMA1996/PI1M/tree/master) .

### generate tfrecord

```shell
python3 create_dataset.py --input_csv <path/to/PI1M.csv> --output_tfrecord <path/to/tfrecord/file>
```

## train feature extractor

```shell
python3 transfer_learning.py --dataset <path/to/tfrecord/file> --ckpt <path/to/place/ckpt>
```
