# fashion-recommender
Fashion Recommendation System with RAG

### Features
- Deep integration of dynamic recommendation system and RAG
- Support natural language query parsing
- Automatic detection of image paths
- Configurable weight strategy

### Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
./scripts/setup_data.sh
```

2. Run the example：
```bash
from core.data_processor import DataProcessor
from core.recommender import EnhancedRecommender
from core.rag_system import FashionRAG

# 初始化系统
processor = DataProcessor("data/fashion-product-images-small/myntradataset")
df = processor.load_and_clean()
rag = FashionRAG()
recommender = EnhancedRecommender(df, rag=rag)

# 获取推荐
results = recommender.recommend("适合夏季派对的复古连衣裙")
```


### **Key optimization points**
1. **Modular design**: Separate data processing, recommendation logic, and RAG system into independent modules
2. **Configuration management**: Use YAML files to uniformly manage path configuration
3. **Automated deployment**: Provide one-click data download script
4. **Scalability**: Reserve weight adjustment interface and RAG extension interface

Before deployment, ensure:
1. Configure API key in Kaggle account to download data set
2. Knowledge base files need to be pre-trained or use the provided sample files
3. Image path configuration is consistent with the actual storage location

