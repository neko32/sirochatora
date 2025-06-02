from libs.sirochatora.sirochatora import Sirochatora
from libs.sirochatora.rag.rag import GithubRAG
from libs.sirochatora.util.siroutil import ConfJsonLoader

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import Client

from os import getenv, environ

def main():

    conf:ConfJsonLoader = ConfJsonLoader("sirochatora/testdatagen_conf.json")
    environ["LANGSMITH_API_KEY"] = conf._conf["LANGSMITH_API_KEY"]
    environ["LANGCHAIN_PROJECT"] = conf._conf["LANGCHAIN_PROJECT"]

    dataset_name = "langchain_git_test_data"

    sc:Sirochatora = Sirochatora(temperature=0.)
    embedding_model_local_dir = getenv("LOCAL_EMBEDDING_MODEL_DIR")
    if embedding_model_local_dir is None:
        raise RuntimeError("LOCAL_EMBEDDING_MODEL_DIR must be set")

    embeddings = HuggingFaceEmbeddings(model_name = f"{embedding_model_local_dir}/multilingual-e5-base")

    task_name = "nekogit_langchain_sample"
    gitlag = GithubRAG(src = "https://github.com/langchain-ai/langchain", filter_cond = ".mdx")
    gitlag.set_branch_name("master")
    gitlag.fetch(task_name)

    llm = LangchainLLMWrapper(sc._llm)
    gen_embedding = LangchainEmbeddingsWrapper(embeddings)

    gen = TestsetGenerator(llm = llm, embedding_model = gen_embedding)

    dataset = gen.generate_with_langchain_docs(gitlag._docs[task_name], testset_size = 4)

    data_frames = dataset.to_pandas()
    print(data_frames)

    cl = Client()
    
    if cl.has_dataset(dataset_name = dataset_name):
        cl.delete_dataset(dataset_name = dataset_name)
    
    ls_dataset = cl.create_dataset(dataset_name = dataset_name)

    ipt = []
    out = []
    meta = []

    for test_rec in dataset:
        ipt.append(
            {
                "question": test_rec.eval_sample.user_input,
            }
        )
        out.append(
            {
                "context": None,
                "ground_truth": test_rec.eval_sample.reference
            }
        )
        meta.append(
            {
                "evolution_type": test_rec.synthesizer_name
            }
        )
    cl.create_examples(
        inputs = ipt,
        outputs = out,
        metadata = meta,
        dataset_id = dataset.run_id
    )


if __name__ == "__main__":
    main()