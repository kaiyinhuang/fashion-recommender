#!/bin/bash

# 下载Kaggle数据集
pip install kaggle
mkdir -p data
kaggle datasets download paramaggarwal/fashion-product-images-small -p data/
unzip data/fashion-product-images-small.zip -d data/

# 下载知识库文件
wget https://example.com/fashion_knowledge.index -P data/
wget https://example.com/fashion_knowledge.json -P data/
