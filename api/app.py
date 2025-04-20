from fastapi import FastAPI
from pydantic import BaseModel
from core.recommender import EnhancedRecommender

app = FastAPI(
    title="Fashion Recommendation API",
    description="提供智能服装推荐服务的REST API",
    version="1.0.0"
)

class RecommendationRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/v1/recommend")
async def get_recommendations(request: RecommendationRequest):
    """
    获取服装推荐结果
    
    - query: 自然语言查询（如"适合夏季的波西米亚风格长裙"）
    - top_k: 返回推荐数量（默认5条）
    """
    return recommender.recommend(request.query, request.top_k).to_dict(orient="records")
