# Introduction

## install prerequisite packages

```shell
pip3 install -r requirements.txt
```

if version of python is over 3.9 (included), please manually update np.int to np.int32 at

- /home/breadbread1984/envs/tf2/lib/python3.11/site-packages/networkx/readwrite/graphml.py:346
- /home/breadbread1984/envs/tf2/lib/python3.11/site-packages/networkx/readwrite/gexf.py:220

## download polymer dataset

download the dataset from [PI1M](https://github.com/RUIMINMA1996/PI1M/tree/master) .
