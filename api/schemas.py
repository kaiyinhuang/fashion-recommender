from pydantic import BaseModel

class RecommendationResult(BaseModel):
    item_id: str
    article_type: str
    base_colors: list[str]
    score: float
    image_url: str
    details: dict
