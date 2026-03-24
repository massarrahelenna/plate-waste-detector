import streamlit as st
from roboflow import Roboflow
from PIL import Image
import numpy as np
import cv2
from collections import Counter
import matplotlib.pyplot as plt

# Inicializar modelo Roboflow
rf = Roboflow(api_key="0F8rtF0Q8QsBh7H2d1Jv")
project = rf.workspace("processamento-de-imagem-aula").project("my-first-project-lzc3k")
model = project.version(2).model

# Função de pré-processamento
def enhance_detection(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (640, 640))
    return image

# Função de predição
def predict(image):
    processed_img = enhance_detection(image)
    prediction = model.predict(processed_img, confidence=30, overlap=25).json()
    return prediction

# Configurações da página
st.set_page_config(page_title="Desperdício Zero", page_icon="🍽️", layout="wide")

st.markdown("""
    <style>
        div[data-baseweb="radio"] {
            display: flex;
            justify-content: center;
            gap: 30px;
        }
        label[data-baseweb="radio"] {
            font-size: 18px;
            color: #4CAF50;
            font-weight: bold;
            cursor: pointer;
        }
        section[data-testid="stSidebar"] {
            background-color: #f0f0f0;
        }
    </style>

    <div style="text-align: center; font-size: 30px; font-weight: bold;">Desperdício Zero</div>
""", unsafe_allow_html=True)

# Menu com st.radio estilizado como cabeçalho
page = st.radio(
    "Navegação",
    ["Início", "Registros", "Relatórios", "Configurações"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("</div><hr>", unsafe_allow_html=True)
if page == "Início":
    # Página de Início
    st.subheader("Bem-vindo ao Desperdício Zero")
    st.write("O aplicativo que ajuda restaurantes a reduzirem o desperdício de alimentos com base em dados reais.")

    # Quem Somos
    st.markdown("""
        <section style="background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 20px;">
            <h3 style="color: #4CAF50; font-size: 24px; font-weight: 600; text-align: center;">Quem Somos</h3>
            <p>O <strong>Desperdício Zero</strong> nasceu com a missão de revolucionar a gestão de resíduos alimentares em restaurantes. Utilizamos tecnologia de ponta para promover sustentabilidade e eficiência.
Com nosso sistema de visão computacional, analisamos imagens de pratos pós-refeição enviadas por sua equipe, identificando padrões de desperdício. Isso permite ajustes precisos no porcionamento e uma gestão mais inteligente de insumos.
Somos uma equipe apaixonada por inovação, sustentabilidade e impacto social. Nossa visão é criar um futuro onde cada refeição seja planejada com consciência, reduzindo custos e o impacto ambiental.
Junte-se a nós e faça parte dessa transformação para um setor de alimentação mais responsável e eficiente.</p>
        </section>
    """, unsafe_allow_html=True)


    st.markdown("</section>", unsafe_allow_html=True)

elif page == "Registros":
    
    # Processamento de Imagens
    st.subheader("Processamento de Imagens")
    st.write("Envie uma imagem para processamento:")

    uploaded_file = st.file_uploader("Escolha uma imagem JPG", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem Carregada", use_column_width=True)
        image_np = np.array(image)

        if st.button('Processar Imagem'):
            with st.spinner('Processando...'):
                prediction = predict(image_np)
                st.success('Análise Concluída!')

                if prediction['predictions']:
                    st.write("Predições detectadas:")
                    for pred in prediction['predictions']:
                        st.write(f"{pred['class']} - Confiança: {pred['confidence']:.2f}")
                else:
                    st.write("Nenhuma predição detectada.")


elif page == "Relatórios":
    # Página de Relatórios
    st.subheader("Relatórios de Desperdício de Alimentos")
    st.write("Aqui você pode ver os relatórios detalhados sobre desperdícios.")
    
    annotations_path = "./_annotations.txt"
    classes_path = "./_classes.txt"
    
    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f]

    # 2. Contar frequência das classes nas anotações
    food_counts = Counter()

    with open(annotations_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            boxes = parts[1:]  # Ignora o nome da imagem
            for box in boxes:
                try:
                    *coords, class_id = box.split(",")
                    food_counts[int(class_id)] += 1
                except ValueError:
                    continue  # Ignora erros de formatação

    # 3. Mostrar os resultados
    st.write("🍽️ Frequência de alimentos detectados nos pratos:")
    for class_id, count in food_counts.most_common():
        st.write(f"{class_names[class_id]}: {count}")

    # 4. Visualização com gráfico de barras
    labels = [class_names[cid] for cid, _ in food_counts.most_common()]
    values = [count for _, count in food_counts.most_common()]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='tomato')
    plt.xlabel("Alimentos")
    plt.ylabel("Quantidade detectada")
    plt.title("Frequência de Alimentos Detectados (possível desperdício)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Exibir o gráfico no Streamlit
    st.pyplot(plt)
    # Página de Relatórios
    st.subheader("Relatórios de Desperdício de Alimentos")
    st.write("Aqui você pode ver os relatórios detalhados sobre desperdícios.")
    
    
    annotations_path = "./_annotations.txt"
    classes_path = "./_classes.txt"
    
    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f]

    # 2. Contar frequência das classes nas anotações
    food_counts = Counter()

    with open(annotations_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            boxes = parts[1:]  # Ignora o nome da imagem
            for box in boxes:
                try:
                    *coords, class_id = box.split(",")
                    food_counts[int(class_id)] += 1
                except ValueError:
                    continue  # Ignora erros de formatação

    # 3. Mostrar os resultados
    print("🍽️ Frequência de alimentos detectados nos pratos:")
    for class_id, count in food_counts.most_common():
        print(f"{class_names[class_id]}: {count}")

    # 4. (Opcional) Visualização com gráfico de barras
    labels = [class_names[cid] for cid, _ in food_counts.most_common()]
    values = [count for _, count in food_counts.most_common()]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='tomato')
    plt.xlabel("Alimentos")
    plt.ylabel("Quantidade detectada")
    plt.title("Frequência de Alimentos Detectados (possível desperdício)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


elif page == "Configurações":
    # Página de Configurações
    st.subheader("Configurações do Sistema")
    st.write("Aqui você pode ajustar as configurações do sistema de reconhecimento e análise.")


# Rodapé fixo
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            color: gray;
            font-size: 12px;
            padding: 10px;
            background-color: white;
            z-index: 100;
        }
    </style>
    <div class="footer">
        &copy; 2025 Desperdício Zero. Todos os direitos reservados.
    </div>
""", unsafe_allow_html=True)

