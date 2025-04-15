import pandas as pd
from fuzzywuzzy import fuzz
from .rag_system import FashionRAG

class EnhancedRecommender:
    def __init__(self, df, rag=None, image_base_path=None):
        self.df = df
        self.rag = rag
        self.image_base_path = image_base_path
        self.weights = {"color":0.4, "type":0.3, "usage":0.3}
        
    def _get_image_path(self, item_id):
        """动态获取图片路径"""
        for ext in ['.jpg', '.png']:
            path = f"{self.image_base_path}/{item_id}{ext}"
            if os.path.exists(path):
                return path
        return None
    
    def recommend(self, query, top_k=5):
        """整合RAG的推荐入口"""
        # RAG参数解析
        rag_params = self.rag.parse_query(query) if self.rag else {}
        
        # 动态过滤
        candidates = self.df.copy()
        if rag_params.get("colors"):
            candidates = candidates[candidates["baseColour"].apply(
                lambda x: len(set(x) & set(rag_params["colors"])) > 0
            )]
        
        # 相似度计算
        candidates["score"] = candidates.apply(
            lambda row: self._calculate_score(row, rag_params), axis=1
        )
        
        # 生成结果
        return candidates.sort_values("score", ascending=False).head(top_k)
    
    def _calculate_score(self, row, params):
        """动态权重评分"""
        score = 0
        # 颜色匹配
        color_overlap = len(set(row["baseColour"]) & set(params.get("colors", [])))
        score += self.weights["color"] * (color_overlap / max(1, len(params.get("colors", []))))
        # 类型模糊匹配
        type_sim = fuzz.partial_ratio(row["articleType"], params.get("type", "")) / 100
        score += self.weights["type"] * type_sim
        return score
