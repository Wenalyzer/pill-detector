from PIL import Image
import io

def save_image_to_base64(image: Image.Image) -> str:
    """將圖片轉為 base64"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    import base64
    return base64.b64encode(buffer.getvalue()).decode()