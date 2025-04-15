import unittest
import pandas as pd
from core import DataProcessor, EnhancedRecommender
from core.rag_system import FashionRAG

class TestRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 初始化测试数据
        cls.df = pd.DataFrame({
            "id": [1, 2, 3],
            "articleType": ["dress", "jeans", "shirt"],
            "baseColour": [["red"], ["blue"], ["white"]]
        })
        cls.rag = FashionRAG()
        cls.recommender = EnhancedRecommender(cls.df, rag=cls.rag)

    def test_basic_recommendation(self):
        # 测试基础推荐逻辑
        results = self.recommender.recommend("红色连衣裙")
        self.assertGreater(len(results), 0)
        self.assertEqual(results.iloc[0]["articleType"], "dress")

    def test_image_path_validation(self):
        # 测试图片路径生成
        path = ImageUtils.validate_image_path("/mock/path", 1)
        self.assertIsNone(path)  # 假设路径不存在

if __name__ == "__main__":
    unittest.main()
