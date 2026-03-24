from roboflow import Roboflow

# Insira sua chave de API
rf = Roboflow(api_key="0F8rtF0Q8QsBh7H2d1Jv")

# Acesse o projeto e versão correta
project = rf.workspace("processamento-de-imagem-aula").project("my-first-project-lzc3k")
version = project.version(4)

# Baixe o dataset no formato YOLOv5
dataset = version.download("yolov5")
