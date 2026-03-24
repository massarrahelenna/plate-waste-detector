"""
detector.py — Motor de inferência e análise do Plate Waste Detector.

Responsabilidades:
  - Pré-processar imagens para o modelo YOLO
  - Calcular área de pixels de cada detecção
  - Estimar peso (g) usando fatores de densidade do dados.json
  - Retornar lista estruturada de alimentos detectados
"""

import json
import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

# ──────────────────────────────────────────────
# Caminhos
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DADOS_PATH = BASE_DIR / "data" / "dados.json"


# ──────────────────────────────────────────────
# Estrutura de resultado
# ──────────────────────────────────────────────
@dataclass
class AlimentoDetectado:
    """Representa um alimento identificado em uma imagem."""
    classe: str                    # nome interno (ex: "arroz")
    nome_display: str              # nome amigável (ex: "Arroz")
    confianca: float               # 0.0 a 1.0
    area_pixels: int               # área da bounding box em pixels²
    peso_estimado_g: float         # peso estimado em gramas
    densidade_usada: float         # fator de densidade aplicado
    aviso_confianca: bool = False  # True se confiança < 0.5

    def to_dict(self) -> dict:
        return {
            "classe": self.classe,
            "nome_display": self.nome_display,
            "confianca": round(self.confianca, 4),
            "area_pixels": self.area_pixels,
            "peso_estimado_g": round(self.peso_estimado_g, 1),
            "aviso_confianca": self.aviso_confianca,
        }


# ──────────────────────────────────────────────
# Carregamento de configuração
# ──────────────────────────────────────────────
def carregar_dados(path: Path = DADOS_PATH) -> dict:
    """
    Carrega o dados.json com os fatores de densidade.
    Lança FileNotFoundError se o arquivo não existir.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: {path}\n"
            "Certifique-se de que 'data/dados.json' existe no projeto."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Pré-processamento
# ──────────────────────────────────────────────
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Converte RGB → BGR e redimensiona para 640×640 (padrão YOLO).

    Args:
        image: Array numpy em formato RGB (vindo do PIL/Streamlit).

    Returns:
        Array numpy em BGR 640×640.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.resize(image_bgr, (640, 640))


# ──────────────────────────────────────────────
# Cálculo de área
# ──────────────────────────────────────────────
def calcular_area_pixels(pred: dict) -> int:
    """
    Calcula a área da bounding box de uma predição Roboflow.

    A API retorna: x, y (centro), width, height — todos em pixels.

    Args:
        pred: Dicionário de predição individual do Roboflow.

    Returns:
        Área em pixels² (inteiro).
    """
    largura = pred.get("width", 0)
    altura = pred.get("height", 0)
    return int(largura * altura)


# ──────────────────────────────────────────────
# Estimativa de peso
# ──────────────────────────────────────────────
def estimar_peso(area_pixels: int, densidade_g_por_pixel: float) -> float:
    """
    Converte área de pixels em gramas usando o fator de densidade.

    Fórmula: Peso (g) = Área (px²) × Densidade (g/px²)

    O fator de densidade deve ser calibrado experimentalmente
    usando uma balança de precisão com porções reais.

    Args:
        area_pixels: Área da bounding box em pixels².
        densidade_g_por_pixel: Fator de conversão g/px² do dados.json.

    Returns:
        Peso estimado em gramas.
    """
    return area_pixels * densidade_g_por_pixel


# ──────────────────────────────────────────────
# Função principal
# ──────────────────────────────────────────────
def process_analysis(predictions: dict) -> list[AlimentoDetectado]:
    """
    Transforma as predições brutas do Roboflow em insights de desperdício.

    Para cada detecção:
      1. Busca o fator de densidade no dados.json pelo nome da classe.
      2. Calcula a área da bounding box.
      3. Estima o peso em gramas.
      4. Sinaliza detecções com confiança baixa (< 0.5).

    Args:
        predictions: JSON retornado por model.predict(...).json()
                     Formato esperado: {"predictions": [...]}

    Returns:
        Lista de AlimentoDetectado, ordenada por peso estimado (maior primeiro).
        Retorna lista vazia se não houver predições.
    """
    dados = carregar_dados()
    config_alimentos = dados.get("alimentos", {})

    preds = predictions.get("predictions", [])
    if not preds:
        return []

    resultados: list[AlimentoDetectado] = []

    for pred in preds:
        classe = pred.get("class", "desconhecido").lower()
        confianca = float(pred.get("confidence", 0.0))

        # Busca configuração do alimento; usa fallback se não encontrado
        config = config_alimentos.get(classe)
        if config is None:
            # Alimento não cadastrado em dados.json — usa densidade genérica
            nome_display = classe.capitalize()
            densidade = 0.07  # fallback conservador
        else:
            nome_display = config.get("nome_display", classe.capitalize())
            densidade = float(config.get("densidade_g_por_pixel", 0.07))

        area = calcular_area_pixels(pred)
        peso = estimar_peso(area, densidade)

        resultados.append(AlimentoDetectado(
            classe=classe,
            nome_display=nome_display,
            confianca=confianca,
            area_pixels=area,
            peso_estimado_g=peso,
            densidade_usada=densidade,
            aviso_confianca=confianca < 0.5,
        ))

    # Ordena por peso estimado, do maior para o menor
    resultados.sort(key=lambda a: a.peso_estimado_g, reverse=True)
    return resultados


def resumo_total(alimentos: list[AlimentoDetectado]) -> dict:
    """
    Gera um resumo agregado a partir da lista de alimentos detectados.

    Args:
        alimentos: Lista retornada por process_analysis().

    Returns:
        Dicionário com:
          - total_g: peso total estimado de desperdício
          - por_alimento: {nome_display: peso_g} de cada item
          - qtd_deteccoes: número total de detecções
          - tem_avisos: True se alguma detecção tem confiança baixa
    """
    if not alimentos:
        return {
            "total_g": 0.0,
            "por_alimento": {},
            "qtd_deteccoes": 0,
            "tem_avisos": False,
        }

    por_alimento: dict[str, float] = {}
    for a in alimentos:
        por_alimento[a.nome_display] = round(
            por_alimento.get(a.nome_display, 0.0) + a.peso_estimado_g, 1
        )

    return {
        "total_g": round(sum(a.peso_estimado_g for a in alimentos), 1),
        "por_alimento": por_alimento,
        "qtd_deteccoes": len(alimentos),
        "tem_avisos": any(a.aviso_confianca for a in alimentos),
    }