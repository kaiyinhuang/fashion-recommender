import os
from PIL import Image
from IPython.display import display

class ImageUtils:
    @staticmethod
    def validate_image_path(base_path, item_id):
        """验证图片路径是否存在"""
        for ext in ['.jpg', '.png', '.jpeg']:
            path = os.path.join(base_path, f"{item_id}{ext}")
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def display_image(image_path, resize=(200,200)):
        """显示图片（支持Jupyter环境）"""
        if image_path:
            img = Image.open(image_path)
            img.thumbnail(resize)
            display(img)
            return True
        return False

    @staticmethod
    def clean_text(text):
        """文本清洗工具"""
        return text.lower().strip().replace('\n', ' ')
