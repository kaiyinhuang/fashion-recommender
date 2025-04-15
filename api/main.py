from fastapi import FastAPI
from pydantic import BaseModel
from core.recommender import EnhancedRecommender
from core.data_processor import DataProcessor
import os

app = FastAPI()

# 初始化推荐系统
data_path = os.path.join("data", "fashion-product-images-small", "myntradataset")
processor = DataProcessor(data_path)
df = processor.load_and_clean()
recommender = EnhancedRecommender(df)

class RecommendationRequest(BaseModel):
    baseColor: list[str]
    gender: str
    season: str
    usage: str
    articleType: str

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    # 转换参数格式
    params = {
        "baseColour": request.baseColor,
        "gender": request.gender.lower(),
        "season": request.season.lower(),
        "usage": request.usage.lower(),
        "articleType": request.articleType.lower()
    }
    
    # 调用推荐系统
    results = recommender.dynamic_recommend(params)
    
    # 构造响应
    return {
        "recommendations": [
            {
                "id": item["id"],
                "type": item["articleType"],
                "colors": item["baseColour"],
                "imageUrl": f"http://your-server/images/{item['id']}.jpg"  # 图片URL示例
            }
            for item in results["recommendations"]
        ],
        "explanations": results["explanations"]
    }
