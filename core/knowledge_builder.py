import pandas as pd
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from deepseek_api import DeepSeekAPI
from pathlib import Path

class KnowledgeBuilder:
    def __init__(self, config_path="configs/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.encoder = SentenceTransformer(self.config["encoder_model"])
        
    def _load_config(self, path):
        # 加载配置逻辑
        return {
            "encoder_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "index_dim": 384
        }

    def build_from_csv(self, input_path, output_dir="knowledge"):
        """从原始对话构建知识库"""
        # 数据清洗与转换
        df = pd.read_csv(input_path)
        structured_data = self._process_with_llm(df)
        
        # 向量索引构建
        embeddings = self._generate_embeddings(structured_data)
        index = self._build_faiss_index(embeddings)
        
        # 保存结果
        self._save_knowledge(structured_data, index, output_dir)
        return structured_data


    DEEPSEEK_API_KEY = "your_api_key_here"
    MAX_RETRIES = 3  # API调用失败重试次数
    

    def clean_message(text):
        # 修复常见拼写错误
        corrections = {
            r"Stealth\s*dress": "Sheath Dress",
            r"Blender": "混纺面料",
            r"linewr": "亚麻",
            r"wisawake": "洗涤"
        }
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 统一格式符号
        text = re.sub(r":--(.*?)--:", r"【\1】", text)  # 中文标点标准化
        return text
    

    def process_with_deepseek(text):
        prompt = f"""请将以下英文服装咨询对话转换为结构化中文数据：
    
                      原始对话：
                      {text}
                      
                      要求：
                      1. 所有内容必须使用中文
                      2. 提取字段包括：
                         - 用户需求（1-2句概括）
                         - 推荐款式（列表）
                         - 面料建议（列表）
                         - 关键特征（字典：长度/领口/袖长）
                         - 相关产品（列表：名称+关键属性）
                         - 场景标签（至少3个分类标签）
                      
                      输出格式（严格JSON格式，不要注释）：
                      {{
                          "用户需求": "",
                          "推荐款式": [],
                          "面料建议": [],
                          "关键特征": {{"长度":"", "领口":"", "袖长":""}},
                          "相关产品": [{{"名称":"", "颜色":"", "材质":""}}],
                          "场景标签": []
                      }}"""
    
        for _ in range(MAX_RETRIES):
            try:
                response = DeepSeekAPI.chat(
                    api_key=DEEPSEEK_API_KEY,
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                cleaned_response = response.replace("，", ",").replace("：", ":")  # 统一符号
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                print("JSON解析失败，尝试重新生成...")
                continue
            except Exception as e:
                print(f"API调用失败：{str(e)}")
                continue
        return {}
    

    # 读取原始数据
    df = pd.read_csv("input.csv")  # 列名应为message
    
    # 清洗数据
    df["cleaned_message"] = df["message"].apply(clean_message)
    
    # 处理前3条示例（实际使用时移除.head(3)）
    results = []
    for idx, msg in df["cleaned_message"].head(3).items():
        print(f"Processing message {idx+1}...")
        result = process_with_deepseek(msg)
        results.append(result)
        print(f"Result: {json.dumps(result, ensure_ascii=False)}")
    

    structured_data = []
    for item in results:
        # 处理嵌套结构
        products = []
        for p in item.get("相关产品", []):
            products.append(f"{p.get('名称','')}（颜色：{p.get('颜色','')}，材质：{p.get('材质','')}）")
        
        structured_data.append({
            "用户需求": item.get("用户需求", ""),
            "推荐款式": "；".join(item.get("推荐款式", [])),
            "适用面料": "；".join(item.get("面料建议", [])),
            "服装长度": item.get("关键特征", {}).get("长度", ""),
            "领口设计": item.get("关键特征", {}).get("领口", ""),
            "袖长类型": item.get("关键特征", {}).get("袖长", ""),
            "推荐产品": "；".join(products),
            "适用场景": "；".join(item.get("场景标签", []))
        })
    

    final_df = pd.DataFrame(structured_data)
    final_df.to_csv("中文知识库.csv", index=False, encoding='utf_8_sig')  # 保证中文编码
  
    def _generate_embeddings(self, data):
        texts = data.apply(lambda x: f"{x['用户需求']} {x['场景标签']}", axis=1)
        return self.encoder.encode(texts.tolist())

    def _build_faiss_index(self, embeddings):
        index = faiss.IndexFlatL2(self.config["index_dim"])
        index.add(np.array(embeddings).astype("float32"))
        return index

    def _save_knowledge(self, data, index, output_dir):
        Path(output_dir).mkdir(exist_ok=True)
        data.to_csv(f"{output_dir}/structured_kb.csv", index=False)
        faiss.write_index(index, f"{output_dir}/kb_index.index")
