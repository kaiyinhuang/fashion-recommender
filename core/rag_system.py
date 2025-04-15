import faiss
import json
from sentence_transformers import SentenceTransformer

class FashionRAG:
    def __init__(self, index_path="data/fashion_knowledge.index", 
                 knowledge_path="data/fashion_knowledge.json"):
        with open(knowledge_path) as f:
            self.knowledge = json.load(f)
        self.index = faiss.read_index(index_path)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def parse_query(self, query):
        """自然语言解析为推荐参数"""
        # 向量化查询
        query_embed = self.encoder.encode([query])
        _, indices = self.index.search(query_embed, 2)
        
        # 提取参数
        params = {"colors": [], "types": []}
        for idx in indices[0]:
            item = self.knowledge[idx]
            params["colors"].extend(item.get("colors", []))
            params["types"].extend(item.get("garments", []))
        return params
