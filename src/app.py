import os
import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    st.error("⚠️ ROBOFLOW_API_KEY não encontrada. Crie um arquivo .env com a variável.")
    st.stop()

BASE_DIR = Path(__file__).parent.parent
ANNOTATIONS_PATH = BASE_DIR / "dataset" / "_annotations.txt"
CLASSES_PATH = BASE_DIR / "dataset" / "_classes.txt"

@st.cache_resource
def load_model():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("processamento-de-imagem-aula").project("my-first-project-lzc3k")
    return project.version(2).model

model = load_model()

def preprocess_image(image: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.resize(image_bgr, (640, 640))

def predict(image: np.ndarray) -> dict:
    processed = preprocess_image(image)
    return model.predict(processed, confidence=30, overlap=25).json()

def draw_detections(image: Image.Image, predictions: dict) -> Image.Image:
    """Desenha bounding boxes sobre a imagem original."""
    img = image.copy().resize((640, 640))
    draw = ImageDraw.Draw(img)

    CORES = {
        "high": "#4CAF50",
        "mid": "#FFC107",
        "low": "#F44336",
    }

    preds = predictions.get("predictions", [])
    for pred in preds:
        x = pred.get("x", 0)
        y = pred.get("y", 0)
        w = pred.get("width", 0)
        h = pred.get("height", 0)
        label = pred.get("class", "?")
        conf = float(pred.get("confidence", 0))

        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2

        cor = CORES["high"] if conf >= 0.7 else CORES["mid"] if conf >= 0.5 else CORES["low"]
        draw.rectangle([x1, y1, x2, y2], outline=cor, width=3)
        draw.rectangle([x1, y1 - 18, x1 + len(label) * 8 + 6, y1], fill=cor)
        draw.text((x1 + 3, y1 - 16), f"{label} {conf:.0%}", fill="white")

    return img

def load_class_names(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_food_counts(annotations_path: Path, class_names: list[str]) -> Counter:
    food_counts = Counter()
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            for box in parts[1:]:
                try:
                    *_, class_id = box.split(",")
                    idx = int(class_id)
                    if 0 <= idx < len(class_names):
                        food_counts[idx] += 1
                except (ValueError, IndexError):
                    continue
    return food_counts

def plot_food_counts(food_counts: Counter, class_names: list[str]):
    labels = [class_names[cid] for cid, _ in food_counts.most_common()]
    values = [count for _, count in food_counts.most_common()]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels[::-1], values[::-1], color="#4CAF50")
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_xlabel("Ocorrências no dataset")
    ax.set_title("Frequência de alimentos detectados (dataset de treino)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_deteccao(alimentos):
    """Gráfico de pizza das detecções da imagem processada."""
    nomes = [a.nome_display for a in alimentos]
    pesos = [a.peso_estimado_g for a in alimentos]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(pesos, labels=nomes, autopct="%1.0f%%", startangle=140,
           colors=plt.cm.Set3.colors[:len(nomes)])
    ax.set_title("Distribuição estimada por alimento")
    st.pyplot(fig)
    plt.close(fig)


# ── Layout ──────────────────────────────────────
st.set_page_config(page_title="Desperdício Zero", page_icon="🍽️", layout="wide")

st.markdown("""
    <style>
        div[data-baseweb="radio"] { display: flex; justify-content: center; gap: 30px; }
        label[data-baseweb="radio"] { font-size: 18px; color: #4CAF50; font-weight: bold; cursor: pointer; }
        section[data-testid="stSidebar"] { background-color: #f0f0f0; }
        .footer { position: fixed; bottom: 0; width: 100%; text-align: center;
                  color: gray; font-size: 12px; padding: 10px;
                  background-color: white; z-index: 100; }
        .metric-card { background:#fff; border-radius:8px; padding:16px;
                       box-shadow:0 2px 4px rgba(0,0,0,0.08); text-align:center; }
    </style>
    <div style="text-align: center; font-size: 30px; font-weight: bold;">Desperdício Zero</div>
""", unsafe_allow_html=True)

page = st.radio(
    "Navegação",
    ["Início", "Registros", "Relatórios", "Configurações"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("<hr>", unsafe_allow_html=True)


# ── Início ───────────────────────────────────────
if page == "Início":
    st.subheader("Bem-vindo ao Desperdício Zero")
    st.write("O aplicativo que ajuda restaurantes a reduzirem o desperdício de alimentos com base em dados reais.")

    col1, col2, col3 = st.columns(3)
    col1.markdown("""
        <div class="metric-card">
            <h2>📷</h2>
            <b>Upload de imagens</b>
            <p style="color:gray;font-size:13px">Envie fotos de pratos para análise instantânea</p>
        </div>
    """, unsafe_allow_html=True)
    col2.markdown("""
        <div class="metric-card">
            <h2>🧠</h2>
            <b>IA de visão computacional</b>
            <p style="color:gray;font-size:13px">Modelo YOLOv8 detecta 18 categorias de alimentos</p>
        </div>
    """, unsafe_allow_html=True)
    col3.markdown("""
        <div class="metric-card">
            <h2>📊</h2>
            <b>Relatórios de desperdício</b>
            <p style="color:gray;font-size:13px">Visualize padrões e tome decisões baseadas em dados</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <section style="background-color:#fff;padding:20px;border-radius:8px;
                        box-shadow:0 4px 6px rgba(0,0,0,0.1);margin-top:20px;">
            <h3 style="color:#4CAF50;font-size:24px;font-weight:600;text-align:center;">Quem Somos</h3>
            <p>O <strong>Desperdício Zero</strong> nasceu com a missão de revolucionar a gestão de resíduos
            alimentares em restaurantes. Utilizamos tecnologia de ponta para promover sustentabilidade e
            eficiência. Com nosso sistema de visão computacional, analisamos imagens de pratos pós-refeição,
            identificando padrões de desperdício e permitindo ajustes precisos no porcionamento.</p>
        </section>
    """, unsafe_allow_html=True)


# ── Registros ────────────────────────────────────
elif page == "Registros":
    st.subheader("Processamento de Imagens")

    conf_threshold = st.session_state.get("conf_threshold", 30)
    overlap_threshold = st.session_state.get("overlap_threshold", 25)

    uploaded_file = st.file_uploader("Envie uma imagem do prato", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col_img, col_result = st.columns(2)

        with col_img:
            st.markdown("**Imagem original**")
            st.image(image, use_container_width=True)

        if st.button("Processar Imagem", type="primary"):
            with st.spinner("Analisando imagem..."):
                try:
                    from detector import process_analysis, resumo_total

                    image_np = np.array(image)
                    prediction = predict(image_np)
                    alimentos = process_analysis(prediction)
                    resumo = resumo_total(alimentos)

                    with col_result:
                        st.markdown("**Detecções**")
                        img_com_boxes = draw_detections(image, prediction)
                        st.image(img_com_boxes, use_container_width=True)

                    st.success("Análise concluída!")

                    if not alimentos:
                        st.warning("Nenhuma detecção encontrada. Tente com outra imagem ou melhore a iluminação.")
                    else:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Detecções", resumo["qtd_deteccoes"])
                        m2.metric("Desperdício estimado", f"{resumo['total_g']} g")
                        m3.metric("Avisos de baixa confiança", "Sim" if resumo["tem_avisos"] else "Não")

                        st.markdown("---")
                        col_list, col_chart = st.columns([1, 1])

                        with col_list:
                            st.markdown("#### Alimentos detectados")
                            for alimento in alimentos:
                                cor = "🟢" if alimento.confianca >= 0.7 else "🟡" if alimento.confianca >= 0.5 else "🔴"
                                st.write(
                                    f"{cor} **{alimento.nome_display}** — "
                                    f"`{alimento.peso_estimado_g} g` "
                                    f"(confiança: {alimento.confianca:.0%})"
                                )
                                if alimento.aviso_confianca:
                                    st.caption("⚠️ Confiança baixa — considere re-fotografar.")

                        with col_chart:
                            plot_deteccao(alimentos)

                except FileNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Erro ao processar imagem: {e}")


# ── Relatórios ───────────────────────────────────
elif page == "Relatórios":
    st.subheader("Relatórios de Desperdício")

    if not ANNOTATIONS_PATH.exists() or not CLASSES_PATH.exists():
        st.warning(
            f"Arquivos de anotação não encontrados.\n\n"
            f"Esperados em:\n- `{ANNOTATIONS_PATH}`\n- `{CLASSES_PATH}`"
        )
    else:
        try:
            class_names = load_class_names(CLASSES_PATH)
            food_counts = load_food_counts(ANNOTATIONS_PATH, class_names)

            if not food_counts:
                st.info("Nenhum dado de desperdício encontrado nas anotações.")
            else:
                total = sum(food_counts.values())
                st.caption(f"Total de anotações no dataset: **{total}** detecções em {len(food_counts)} classes")

                col_table, col_chart = st.columns([1, 2])

                with col_table:
                    st.markdown("**Ranking de alimentos**")
                    for i, (class_id, count) in enumerate(food_counts.most_common(), 1):
                        pct = count / total * 100
                        st.write(f"{i}. **{class_names[class_id]}** — {count} ({pct:.1f}%)")

                with col_chart:
                    plot_food_counts(food_counts, class_names)

        except Exception as e:
            st.error(f"Erro ao carregar relatórios: {e}")


# ── Configurações ────────────────────────────────
elif page == "Configurações":
    st.subheader("Configurações do Sistema")

    st.markdown("#### Parâmetros de detecção")
    conf = st.slider(
        "Confiança mínima (%)",
        min_value=10, max_value=90,
        value=st.session_state.get("conf_threshold", 30),
        step=5,
        help="Detecções abaixo deste valor são ignoradas."
    )
    overlap = st.slider(
        "Sobreposição máxima (% IoU)",
        min_value=10, max_value=80,
        value=st.session_state.get("overlap_threshold", 25),
        step=5,
        help="Controla o Non-Maximum Suppression (NMS)."
    )

    if st.button("Salvar configurações", type="primary"):
        st.session_state["conf_threshold"] = conf
        st.session_state["overlap_threshold"] = overlap
        st.success("Configurações salvas para esta sessão.")

    st.markdown("---")
    st.markdown("#### Sobre o modelo")
    st.markdown("""
    | Parâmetro | Valor |
    |---|---|
    | Arquitetura | YOLOv8-seg |
    | Classes detectadas | 18 |
    | Resolução de entrada | 640 × 640 px |
    | Plataforma | Roboflow (inferência em nuvem) |
    """)

    st.markdown("---")
    st.markdown("#### Classes suportadas")
    classes = [
        "Maçã", "Miolo de Maçã", "Casca de Maçã", "Osso", "Espinha de Peixe",
        "Pão", "Casca de Ovo", "Peixe", "Carne", "Mexilhão",
        "Concha de Mexilhão", "Macarrão", "Laranja", "Casca de Laranja",
        "Batata", "Arroz", "Tomate", "Vegetais"
    ]
    cols = st.columns(3)
    for i, cls in enumerate(classes):
        cols[i % 3].markdown(f"- {cls}")


st.markdown("""
    <div class="footer">&copy; 2025 Desperdício Zero. Todos os direitos reservados.</div>
""", unsafe_allow_html=True)
