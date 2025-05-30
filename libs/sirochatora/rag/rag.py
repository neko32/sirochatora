from os import getenv
from abc import ABC, abstractmethod
from enum import Enum, auto

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document, BaseDocumentTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.document_loaders import GitLoader, DirectoryLoader

from langchain_text_splitters import CharacterTextSplitter

from langchain_chroma import Chroma

from libs.sirochatora.sirochatora import Sirochatora
from libs.sirochatora.util.siroutil import from_system_message_to_tuple 

class Filter(ABC):
    @abstractmethod
    def filter(self, data:str, filt_cond:str) -> bool:
        pass

class PostfixFileFilter(Filter):
    def filter(self, data:str, filt_cond:str) -> bool:
        return data.endswith(filt_cond)


class Phase(Enum):
    INIT = auto()
    DATA_RETRIEVING_FROM_SRC = auto()
    DATA_TRANSFORMING = auto()
    DATA_EMBEDDING = auto()
    DATA_STORING = auto()
    READY_FOR_CTX = auto()
    DATA_RETRIEVING_FROM_VSTORE = auto()

class RAGFeature(ABC):

    def __init__(self,
                src: str,
                filter_cond: str,
                split_chunk_size,
                split_chunk_overlap,
                embedding_model_name
                ):
        self._src = src
        self._filter_cond = filter_cond
        self._loader = None
        self._filter = None
        self._phases:dict[str, Phase] = {}
        self._split_chunk_overlap:int = split_chunk_overlap
        self._split_chunk_size: int = split_chunk_size
        self._embedding_model_name:str = embedding_model_name

    @abstractmethod
    def run(self, task_name:str) -> None:
        pass
    
    @abstractmethod
    def fetch(self, task_name:str) -> None:
        pass

    @abstractmethod
    def transform(self, task_name:str) -> None:
        pass

    @abstractmethod
    def embed(self, task_name:str, query:str) -> None:
        pass

    @abstractmethod
    def get_data_for_ctx(self, task_name:str, query:str) -> dict[int, str]:
        pass

    @abstractmethod
    def query_with_rag(self, task_name:str, query:str, sc:Sirochatora | None) -> str:
        pass

class LocalStorageRAG(RAGFeature):

    def __init__(self, 
                src: str,
                filter_cond: str,
                split_chunk_size: int = 1000,
                split_chunk_overlap: int = 0,
                embedding_model_name: str = "multilingual-e5-base"
                ):
        super().__init__(src, filter_cond, split_chunk_size, split_chunk_overlap, embedding_model_name)
        self._loader = DirectoryLoader(
            path = self._src, 
            glob = f"**/*{filter_cond}",
            recursive = True, 
            show_progress = True, 
            use_multithreading = True)
        self._task:dict[str, tuple[Phase, Chroma | None]] = {}
        self._docs:dict[str, list[Document]] = {}
        
    def run(self, task_name:str) -> None:
        pass
    
    def fetch(self, task_name:str) -> None:
        self._task[task_name] = (Phase.DATA_RETRIEVING_FROM_SRC, None)
        print(f"task[{task_name}]@fetch: data retriving from local dir {self._src}")
        self._docs[task_name] = self._loader.load()
        print(f"task[{task_name}]@fetch: data retriving done successfully.")
        print(f"task[{task_name}]@fetch: retrieved doc number is {len(self._docs[task_name])}")

    def transform(self, task_name:str) -> None:
        self._task[task_name] = (Phase.DATA_TRANSFORMING, None)
        print(f"task[{task_name}]@transform: transforming docs with chunk size {self._split_chunk_size} with overlap {self._split_chunk_overlap}")
        text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
        self._docs[task_name] = text_splitter.split_documents(self._docs[task_name])
        print(f"task[{task_name}]@transform: transforming docs done successfully.")
        print(f"task[{task_name}]@transform: transformed doc number is {len(self._docs[task_name])}")

    def embed(self, task_name:str, query:str) -> None:
        self._task[task_name] = (Phase.DATA_EMBEDDING, None)
        print(f"task[{task_name}]@embedding: embedding query with model {self._embedding_model_name}..")

        embedding_model_local_dir = getenv("LOCAL_EMBEDDING_MODEL_DIR")
        if embedding_model_local_dir is None:
            raise RuntimeError("LOCAL_EMBEDDING_MODEL_DIR must be set")
        embeddings = HuggingFaceEmbeddings(model_name = f"{embedding_model_local_dir}/{self._embedding_model_name}")
        vec = embeddings.embed_query(query)
        print(f"task[{task_name}]@embedding: embedding query done. result vec tensor: {len(vec)}")
        self._task[task_name] = (Phase.READY_FOR_CTX,  Chroma.from_documents(self._docs[task_name], embeddings))
    
    def get_data_for_ctx(self, task_name:str, query:str) -> dict[int, str]:
        self._task[task_name] = (Phase.DATA_RETRIEVING_FROM_VSTORE, self._task[task_name][1])
        print(f"task[{task_name}]@embedding: querying against vstore..")
        db_candidate = self._task[task_name][1]
        if isinstance(db_candidate, Chroma):
            db: Chroma = db_candidate
            retriever = db.as_retriever()
            ctx_docs = retriever.invoke(query)
        else:
            raise RuntimeError("Chroma vsotre is not ready yet")

        print(f"task[{task_name}]@embedding: querying against vstore done. retrieved ctx docs size is {len(ctx_docs)}")

        buf:dict[int, str] = {}
        for idx, doc in enumerate(ctx_docs):
            buf[idx] = doc.page_content

        self._task[task_name] = (Phase.READY_FOR_CTX, self._task[task_name][1])
        return buf

    def query_with_rag(self, task_name:str, query:str, sc:Sirochatora | None) -> str:
        raise NotImplementedError()

    def get_retriever(self, task_name:str) -> VectorStoreRetriever:

        if self._task[task_name][0] != Phase.READY_FOR_CTX:
            raise RuntimeError("Chroma vsotre is not ready yet")

        db_candidate = self._task[task_name][1]
        if isinstance(db_candidate, Chroma):
            db: Chroma = db_candidate
            return db.as_retriever()
        else:
            raise RuntimeError("Chroma vstore is not properly set")

class TavilyRAG(RAGFeature):
    def __init__(self, 
                src: str,
                filter_cond: str,
                split_chunk_size: int = 1000,
                split_chunk_overlap: int = 0,
                embedding_model_name: str = "multilingual-e5-base"
                ):
        super().__init__(src, filter_cond, split_chunk_size, split_chunk_overlap, embedding_model_name)
        self._task:dict[str, tuple[Phase, Chroma | None]] = {}
        self._docs:dict[str, list[Document]] = {}

    def run(self, task_name:str) -> None:
        pass
    
    def fetch(self, task_name:str) -> None:
        pass

    def transform(self, task_name:str) -> None:
        pass

    def embed(self, task_name:str, query:str) -> None:
        pass

    def get_data_for_ctx(self, task_name:str, query:str) -> dict[int, str]:
        raise NotImplementedError()

    def query_with_rag(self, task_name:str, query:str, sc:Sirochatora | None) -> str:

        if sc is not None and hasattr(sc, "_llm") and sc._llm:
            llm = sc._llm
        else:
            raise RuntimeError("LLM not found")

        self._task[task_name] = (Phase.DATA_RETRIEVING_FROM_SRC, None)
        msg_from_human = '''\
        以下のコンテキストだけを踏まえて質問に解答して下さい。

        コンテキスト:"""
        {context}
        """

        質問: {question}
        '''

        system_msg = None
        if sc is not None and hasattr(sc, "_system_msgs") and sc._system_msgs:
            system_msg = from_system_message_to_tuple(sc._system_msgs[-1])
        else:
            system_msg = ("system", "賢いねこちゃんとして答えてね")

        prompt = ChatPromptTemplate.from_messages([
            system_msg,
            ("human", msg_from_human)
        ])
        retriever = TavilySearchAPIRetriever(k = 3)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

#        return chain.invoke({"question": query})
        return chain.invoke(query)
