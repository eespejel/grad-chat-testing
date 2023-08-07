from typing import Dict, List

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import CohereEmbeddings, OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from rich.console import Console
from termcolor import cprint
from utils import (JSONLLoader, get_cohere_api_key, get_file_path,
                   get_openai_api_key, get_query_from_user, load_config)

console = Console()

with open("hashira/prompt_ai_conversation.txt", "r") as file:
    PROMPT = file.read()

PROMPT_TEMPLATE_CHAT = PromptTemplate(
    input_variables=["history", "input"], template=PROMPT
)


def load_documents(file_path: str) -> List[Dict]:
    """
    Carga los documentos desde un archivo JSONL y los divide en trozos.

    Args:
        file_path (str): Ruta al archivo JSONL.

    Returns:
        Los documentos cargados y divididos en trozos.
    """
    config = load_config()
    metadata_keys = config["metadata_keys"]
    if metadata_keys is None:
        raise ValueError("metadata_keys must be provided in the configuration")
    loader = JSONLLoader(
        file_path=file_path, page_content_key="texto", metadata_keys=metadata_keys
    )
    data = loader.load()

    # Display message about the number of documents loaded
    cprint(f"Loaded {len(data)} documents.", "green")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["text_splitting"]["chunk_size"],
        chunk_overlap=config["text_splitting"]["chunk_overlap"],
        length_function=len,
    )

    split_data = text_splitter.split_documents(data)

    # Display message about the number of documents after splitting
    cprint(f"Split documents into {len(split_data)} chunks.", "green")

    return split_data


def select_embedding_provider(provider: str, model: str):
    """
    Selecciona el proveedor de embeddings para el chatbot.

    Args:
        provider (str): El proveedor de embeddings. 'openai' o 'cohere'.
        model (str): El modelo a usar para los embeddings.

    Returns:
        El objeto embeddings del proveedor seleccionado.
    """
    if provider.lower() == "openai":
        get_openai_api_key()
        return OpenAIEmbeddings(model=model)
    elif provider.lower() == "cohere":
        get_cohere_api_key()
        return CohereEmbeddings(model=model)
    else:
        raise ValueError(
            f"Proveedor de embedding no compatible: {provider}. Los proveedores admitidos son 'OpenAI' y 'Cohere'."
        )


def get_chroma_db(embeddings, documents, path):
    """
    Obtiene la base de datos Chroma. Crea una nueva o carga una existente.

    Args:
        embeddings: El objeto embeddings a usar.
        documents: Los documentos a indexar en la base de datos.
        path: La ruta a la base de datos Chroma.

    Returns:
        El objeto de la base de datos Chroma.
    """
    config = load_config()
    if config["recreate_chroma_db"]:
        console.print("Recreando Chroma DB...")
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=path,
        )
    else:
        console.print("Cargando Chroma DB existente...")
        return Chroma(persist_directory=path, embedding_function=embeddings)


class hashira():
    def __init__(self) -> None:
        pass
    
    def startup(self) -> bool:
        self.config = load_config()
        self.embeddings = select_embedding_provider(self.config["embeddings_provider"], self.config["embeddings_model"])
        self.documents = load_documents(get_file_path())
        self.vectorstore = get_chroma_db(self.embeddings, self.documents, self.config["chroma_db_name"])
        console.print(f"[green]Documentos {len(self.documents)} cargados.[/green]")
        return True

    def process_query(self, query: str) -> str:
        """
        Procesa una consulta del usuario y genera una respuesta del chatbot.

        Args:
            query (str): La consulta del usuario.
            vectorstore: La base de datos de vectores donde buscar la respuesta.

        Returns:
            La respuesta generada por el chatbot.
        """
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config["document_retrieval"]["k"]})
        memory = VectorStoreRetrieverMemory(retriever=retriever)

        llm = ChatOpenAI(
            model_name=self.config["chat_model"]["model_name"],
            temperature=self.config["chat_model"]["temperature"],
            max_tokens=self.config["chat_model"]["max_tokens"],
        )

        conversation_with_summary = ConversationChain(
            prompt=PROMPT_TEMPLATE_CHAT,
            llm=llm,
            memory=memory,
            verbose=self.config["conversation_chain"]["verbose"],
        )

        c = conversation_with_summary.predict(input=query)
        return c 

    def single_question(self, query):
        """
        Pregunta sencilla
        """
        response = self.process_query(query)
        console.print(f"[red]IA:[/red] {response}")

def hashira_run(a,b) -> None:
    h = hashira()
    if h.startup():
        h.single_question("clima de bogotá")

if __name__ == "__main__":
    hashira_run(None, None)
