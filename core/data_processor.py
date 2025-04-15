import pandas as pd
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load_and_clean(self):
        """数据加载与清洗流程"""
        # Download latest version
        path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")

        df = pd.read_csv(os.path.join(self.data_path, "styles.csv"), on_bad_lines="skip")
        
        # 清洗逻辑
        columns_to_drop = ["masterCategory", "subCategory", "year", "productDisplayName"]  
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # 处理缺失值：删除关键字段缺失的行
        key_columns = ["id", "gender", "articleType", "usage", "baseColour", "season"]
        df = df.dropna(subset=key_columns)

        # 统一文本格式为小写
        text_columns = ["gender", "articleType", "usage", "baseColour", "season"]
        df[text_columns] = df[text_columns].apply(lambda x: x.str.lower())

        df['baseColour'] = df['baseColour'].apply(lambda x: x.split(",") if isinstance(x, str) else x  # 处理字符串或列表)
        df['baseColour'] = df['baseColour'].apply(lambda x: [c.strip() for c in x] if isinstance(x, list) else [])
        return df
