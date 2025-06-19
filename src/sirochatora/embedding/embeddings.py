from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from os import getenv
from hashlib import md5

class Sima:

    def __init__(self,
                session_name: str,
                split_chunk_size: int = 1000,
                split_chunk_overlap: int = 0,
                embedding_model_name: str = "multilingual-e5-base"):
        self._split_chunk_size = split_chunk_size
        self._spilt_chunk_overlap = split_chunk_overlap
        self._embedding_model_name = embedding_model_name
        self._session_name = session_name

        embedding_model_local_dir = getenv("LOCAL_EMBEDDING_MODEL_DIR")
        if embedding_model_local_dir is None:
            raise RuntimeError("LOCAL_EMBEDDING_MODEL_DIR must be set")
        self._embeddings = HuggingFaceEmbeddings(model_name = f"{embedding_model_local_dir}/{self._embedding_model_name}")
        
        indb_dir = getenv("NEKOKAN_INDB_PATH")
        self._persistence_dir = f"{indb_dir}/sirochatora/vector_embedding_db"

        self._vstore = Chroma(
            collection_name = self._session_name,
            embedding_function = self._embeddings,
            persist_directory = self._persistence_dir
        )

    def add(self, doc:str, metadata:dict[str,str]) -> str:
        doc_md5 = md5(doc.encode()).hexdigest()
        doc_obj = Document(page_content = doc, metadata = metadata)
        self._vstore.add_documents(documents = [doc_obj], ids = [doc_md5])
        return doc_md5

    def add_bulk(self, docs:list[str], metas:list[dict[str, str]]) -> list[str]:
        doc_md5 = [
            md5(doc.encode()).hexdigest() for doc in docs
        ]
        doc_objs = [
            Document(page_content = doc, metadata = meta)
            for doc, meta in zip(docs, metas)
        ]
        self._vstore.add_documents(documents = doc_objs, ids = doc_md5)

        return doc_md5

    def similarity_search(self, 
                        query:str, 
                        k:int = 3, 
                        filter:dict[str, str]|None = None):
        rez = self._vstore.similarity_search_with_score(query = query, k = k, filter = filter)
        return [(rez_item.page_content, score) for rez_item,score in rez]
    

    




