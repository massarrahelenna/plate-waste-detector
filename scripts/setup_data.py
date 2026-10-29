import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise EnvironmentError("ROBOFLOW_API_KEY não encontrada. Configure no arquivo .env")

rf = Roboflow(api_key=api_key)
project = rf.workspace("processamento-de-imagem-aula").project("my-first-project-lzc3k")
version = project.version(4)
dataset = version.download("yolov5")
