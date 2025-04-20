import pytest
from core.rag_system import EnhancedFashionRAG

class TestRAGSystem:
    @pytest.fixture
    def rag_system(self):
        return EnhancedFashionRAG(config_path="configs/model_config.yaml")

    def test_query_parsing(self, rag_system):
        test_cases = [
            ("寻找适合小个子的职业装连衣裙", 
             {"colors": ["black", "navy"], "styles": ["A-line"]}),
            ("夏季波西米亚风格长裙",
             {"materials": ["cotton", "linen"], "styles": ["Bohemian"]})
        ]
        
        for query, expected in test_cases:
            params = rag_system.parse_query(query)
            assert all(k in params for k in expected.keys())
