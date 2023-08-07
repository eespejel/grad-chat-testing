import os
import sys

import jsonlines
import yaml
from langchain.schema import Document


class JSONLLoader:
    """
    Cargador de documentos de documentaciones en formato JSONL.

    Args:
        file_path (str): Ruta al archivo JSONL a cargar.
        metadata_keys (list): Lista de claves a extraer del JSONL y añadir a los metadatos.
    """

    def __init__(self, file_path: str, page_content_key: str, metadata_keys: list):
        self.file_path = file_path
        self.page_content_key = page_content_key
        self.metadata_keys = metadata_keys

    def load(self):
        """
        Carga los documentos de la ruta del archivo especificada durante la inicialización.

        Returns:
            Una lista de objetos Document.
        """
        with jsonlines.open(self.file_path) as reader:
            documents = []
            for obj in reader:
                page_content = obj.get(self.page_content_key, "")

                # Only include keys specified in metadata_keys
                # If the value is None, replace it with "desconocido"
                metadata = {
                    key: "desconocido" if obj.get(key) is None else obj.get(key)
                    for key in self.metadata_keys
                }

                documents.append(Document(page_content=page_content, metadata=metadata))
        return documents


def load_config():
    """
    Carga la configuración de la aplicación desde el archivo 'config.yaml'.

    Returns:
        Un diccionario con la configuración de la aplicación.
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root_dir, "config.yaml")) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def get_openai_api_key():
    """
    Obtiene la clave API de OpenAI del entorno. Si no está disponible, detiene la ejecución del programa.

    Returns:
        La clave API de OpenAI.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Por favor crea una variable de ambiente OPENAI_API_KEY.")
        sys.exit()
    return openai_api_key


def get_cohere_api_key():
    """
    Obtiene la clave API de Cohere del entorno. Si no está disponible, solicita al usuario que la ingrese.

    Returns:
        La clave API de Cohere.
    """
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        cohere_api_key = input("Por favor ingresa tu COHERE_API_KEY: ")
    return cohere_api_key


def get_file_path():
    """
    Obtiene la ruta al archivo de base de datos JSONL especificado en la configuración de la aplicación.

    Returns:
        La ruta al archivo de base de datos JSONL.
    """
    config = load_config()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(root_dir, "..")

    return os.path.join(parent_dir, config["jsonl_database_path"])


def get_query_from_user() -> str:
    """
    Solicita una consulta al usuario.

    Returns:
        La consulta ingresada por el usuario.
    """
    try:
        query = input()
        return query
    except EOFError:
        print("Error: Input no esperado. Por favor intenta de nuevo.")
        return get_query_from_user()
