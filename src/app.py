import os
import streamlit as st
from roboflow import Roboflow
from PIL import Image
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

# Caminhos base robustos (funcionam de qualquer diretório)
BASE_DIR = Path(__file__).parent.parent
ANNOTATIONS_PATH = BASE_DIR / "data" / "_annotations.txt"
CLASSES_PATH = BASE_DIR / "data" / "_classes.txt"

@st.cache_resource
def load_model():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("processamento-de-imagem-aula").project("my-first-project-lzc3k")
    return project.version(2).model

model = load_model()

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Converte e redimensiona a imagem para o formato esperado pelo modelo."""
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.resize(image_bgr, (640, 640))

def predict(image: np.ndarray) -> dict:
    """Executa a inferência e retorna as predições."""
    processed = preprocess_image(image)
    return model.predict(processed, confidence=30, overlap=25).json()

def load_class_names(path: Path) -> list[str]:
    """Lê os nomes das classes de um arquivo."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_food_counts(annotations_path: Path, class_names: list[str]) -> Counter:
    """Conta a frequência de cada classe nas anotações."""
    food_counts = Counter()
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            for box in parts[1:]:  # ignora o nome da imagem
                try:
                    *_, class_id = box.split(",")
                    idx = int(class_id)
                    if 0 <= idx < len(class_names):
                        food_counts[idx] += 1
                except (ValueError, IndexError):
                    continue
    return food_counts

def plot_food_counts(food_counts: Counter, class_names: list[str]):
    """Gera e exibe o gráfico de barras no Streamlit."""
    labels = [class_names[cid] for cid, _ in food_counts.most_common()]
    values = [count for _, count in food_counts.most_common()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color="tomato")
    ax.set_xlabel("Alimentos")
    ax.set_ylabel("Quantidade detectada")
    ax.set_title("Frequência de Alimentos Detectados (possível desperdício)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # libera memória

st.set_page_config(page_title="Desperdício Zero", page_icon="🍽️", layout="wide")

st.markdown("""
    <style>
        div[data-baseweb="radio"] { display: flex; justify-content: center; gap: 30px; }
        label[data-baseweb="radio"] { font-size: 18px; color: #4CAF50; font-weight: bold; cursor: pointer; }
        section[data-testid="stSidebar"] { background-color: #f0f0f0; }
        .footer { position: fixed; bottom: 0; width: 100%; text-align: center;
                  color: gray; font-size: 12px; padding: 10px;
                  background-color: white; z-index: 100; }
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

if page == "Início":
    st.subheader("Bem-vindo ao Desperdício Zero")
    st.write("O aplicativo que ajuda restaurantes a reduzirem o desperdício de alimentos com base em dados reais.")
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


elif page == "Registros":
    st.subheader("Processamento de Imagens")
    st.write("Envie uma imagem para processamento:")

    uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagem Carregada", use_column_width=True)
        image_np = np.array(image)

        if st.button("Processar Imagem"):
            with st.spinner("Processando..."):
                try:
                    from detector import process_analysis, resumo_total

                    prediction = predict(image_np)
                    alimentos = process_analysis(prediction)
                    resumo = resumo_total(alimentos)

                    st.success("Análise Concluída!")

                    if not alimentos:
                        st.warning("Nenhuma detecção encontrada. Tente com outra imagem.")
                    else:
                        # ── Resumo geral ──
                        col1, col2 = st.columns(2)
                        col1.metric("Detecções", resumo["qtd_deteccoes"])
                        col2.metric("Desperdício estimado", f"{resumo['total_g']} g")

                        # ── Detalhes por alimento ──
                        st.markdown("#### Alimentos detectados")
                        for alimento in alimentos:
                            cor = "🟢" if alimento.confianca >= 0.7 else "🟡" if alimento.confianca >= 0.5 else "🔴"
                            st.write(
                                f"{cor} **{alimento.nome_display}** — "
                                f"`{alimento.peso_estimado_g} g` estimados "
                                f"(confiança: {alimento.confianca:.0%})"
                            )
                            if alimento.aviso_confianca:
                                st.caption("⚠️ Confiança baixa — considere re-fotografar com melhor iluminação.")

                except FileNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Erro ao processar imagem: {e}")

elif page == "Relatórios":
    st.subheader("Relatórios de Desperdício de Alimentos")

    # Verifica se os arquivos existem antes de abrir
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
                st.write("🍽️ **Frequência de alimentos detectados nos pratos:**")
                for class_id, count in food_counts.most_common():
                    st.write(f"- **{class_names[class_id]}**: {count} ocorrências")

                plot_food_counts(food_counts, class_names)
        except Exception as e:
            st.error(f"Erro ao carregar relatórios: {e}")

elif page == "Configurações":
    st.subheader("Configurações do Sistema")
    st.write("Aqui você pode ajustar as configurações do sistema de reconhecimento e análise.")
    st.info("Em desenvolvimento.")


st.markdown("""
    <div class="footer">&copy; 2025 Desperdício Zero. Todos os direitos reservados.</div>
""", unsafe_allow_html=True)