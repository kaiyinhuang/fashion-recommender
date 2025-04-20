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
bash scripts/build_knowledge.sh
```

```bash
pytest tests/test_rag.py
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

