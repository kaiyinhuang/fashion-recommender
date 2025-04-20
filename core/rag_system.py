import pandas as pd
from .knowledge_builder import KnowledgeBuilder

class EnhancedFashionRAG:
    def __init__(self, config):
        self.knowledge = pd.read_csv(config["knowledge_path"])
        self.index = faiss.read_index(config["index_path"])
        self.encoder = KnowledgeBuilder().encoder
        self.threshold = 0.6  # 相似度阈值

    def parse_query(self, query):
        # 增强版参数解析
        query_embed = self.encoder.encode([query])
        distances, indices = self.index.search(query_embed, 3)
        
        params = {
            "colors": [],
            "styles": [],
            "materials": []
        }
        
        for idx, score in zip(indices[0], distances[0]):
            if score < self.threshold:
                item = self.knowledge.iloc[idx]
                params["colors"].extend(item["baseColour"])
                params["styles"].append(item["推荐款式"])
                params["materials"].append(item["面料建议"])
        
        return self._deduplicate_params(params)

    def _deduplicate_params(self, params):
        # 去重逻辑
        return {
            "colors": list(set(params["colors"])),
            "styles": list(set(params["styles"])),
            "materials": list(set(params["materials"]))
        }
