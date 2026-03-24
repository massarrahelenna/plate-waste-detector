import cv2
import numpy as np

def enhance_image(image):
    """Padroniza a imagem para o modelo."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.resize(image, (640, 640))

def process_analysis(predictions):
    """Transforma as predições em insights de desperdício."""
    # ... toda aquela lógica do seu pandas e contagem de alimentos ...
    return sugestoes