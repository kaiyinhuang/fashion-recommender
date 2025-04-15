#!/bin/bash

# 安装系统依赖
sudo apt-get update && sudo apt-get install -y \
    unzip \
    wget

# 安装Python依赖
pip install -r requirements.txt

# 安装Kaggle CLI（数据下载需要）
pip install kaggle
