from sirochatora.sirochatora import Sirochatora
from sirochatora.rag.rag import LocalStorageRAG, TavilyRAG, RetrievalType, GithubRAG
from sirochatora.util.siroutil import ConfJsonLoader, ctxdict_to_str
from os import environ
import json


def simple_q(sc:Sirochatora):
    sc.append_system_message("Rule 2. 言葉の最後に必ずみゃお！といってね")
    print(sc.query("ねこの外遊びで有名なものは？"))

def simple_template(sc:Sirochatora):
    sc.clear_system()
    sc.add_system_message("入力された素材を使った料理を３つ考える")
    print(sc.query_with_template("{food}", {"food":"ねぎ"}))
    print(sc.query_with_template("{food}", {"food":"しょうが"}))
    print(sc.query_with_template("{food}", {"food":"醤油"}))

def rag_localdoc_simple():
    lc:LocalStorageRAG = LocalStorageRAG(
        "/mnt/d/dataset_for_ml/sanomaru",
        "txt"
    )
    task_name:str = "neko try"
    lc.fetch(task_name)
    lc.transform(task_name)
    lc.embed(task_name, "さのまるについてですが")
    rez = lc.get_data_for_ctx(task_name, "さのまるとトゲトゲダイヤモンド鉄球について聞きたい。")
    print(ctxdict_to_str(rez))
    return rez
    
def rag_localdoc_q_with_ctx(sc:Sirochatora):
    lc:LocalStorageRAG = LocalStorageRAG(
        "/mnt/d/dataset_for_ml/sanomaru",
        "txt"
    )
    task_name:str = "neko try"
    lc.fetch(task_name)
    lc.transform(task_name)
    lc.embed(task_name, "さのまるの手がトゲトゲダイヤモンド鉄球になった件について聞きたい。")
    rez = sc.query_within_context(
        "さのまるはダイヤモンド化して何をしてしまったの？ 起こったことを時系列に５つ教えて",
        lc.get_retriever(task_name)
    )
    print(rez)
    return rez

def ask_zero_cot(sc: Sirochatora):
    rez = sc.query_with_zeroshot_CoT("""クイズだよ
    1. ねことたこが輪になった 
    2. 輪になると光る
    ねことたこはどーなるでしょーか？
""", False)
    print(rez)

def ask_negpos(sc: Sirochatora):
    rez = sc.query_with_negpos_resp("桃太郎が鬼が島に鬼退治に行った結果はどうなるでしょうか?", False)
    print(rez)

def rag_web(sc: Sirochatora):
    rag = TavilyRAG("NA", "NA")
    print(rag.query_with_rag("TAVILY_Q", "ニューヨークの今日の天気は？", sc))

def chat_sample(sc: Sirochatora):
    print(sc.query("ぼくのなまえはさのまるで栃木県佐野市のゆるキャラをやってます"))    
    print(sc.query("ねこちゃんはさのまるに会ったんだね。さのまるってどこのゆるキャラだったかなぁ？"))

def rag_hyde(sc: Sirochatora):
    lc:LocalStorageRAG = LocalStorageRAG(
        "/mnt/d/dataset_for_ml/sanomaru",
        "txt"
    )
    task_name:str = "neko try"
    lc.fetch(task_name)
    lc.transform(task_name)
    lc.embed(task_name, "さのまるの手がトゲトゲダイヤモンド鉄球になった件について聞きたい。")
    rez = sc.query_with_hyde(
        "さのまるはダイヤモンド化して何をしてしまったの？ 起こったことを時系列に５つ教えて",
        lc.get_retriever(task_name)
    )
    print(rez)
    return rez

def rag_multiqueries(sc: Sirochatora):
    lc:LocalStorageRAG = LocalStorageRAG(
        "/mnt/d/dataset_for_ml/sanomaru",
        "txt"
    )
    task_name:str = "neko try"
    lc.fetch(task_name)
    lc.transform(task_name)
    lc.embed(task_name, "さのまるの手がトゲトゲダイヤモンド鉄球になった件について聞きたい。")
    rez = sc.query_with_multiqueries(
        "さのまるはダイヤモンド化して何をしてしまったの？ 起こったことを時系列に５つ教えて",
        lc.get_retriever(task_name)
    )
    print(rez)
    return rez

def rag_bm25(sc: Sirochatora):

    lc:LocalStorageRAG = LocalStorageRAG(
        "/mnt/d/dataset_for_ml/sanomaru",
        "txt"
    )
    task_name:str = "neko try"
    lc.fetch(task_name)
    lc.transform(task_name)
    lc.embed(task_name, "さのまるの手がトゲトゲダイヤモンド鉄球になった件について聞きたい。")
    retv = lc.get_retriever(task_name, RetrievalType.BM25)
    rez = sc.query_with_retriever(
        "さのまるはダイヤモンド化して何をしてしまったの？ 起こったことを時系列に５つ教えて",
        retv
    )
    print(rez)
    return rez

def rag_rerank(sc: Sirochatora):
    lc:LocalStorageRAG = LocalStorageRAG(
        "/mnt/d/dataset_for_ml/sanomaru",
        "txt"
    )
    task_name:str = "neko try"
    lc.fetch(task_name)
    lc.transform(task_name)
    lc.embed(task_name, "さのまるの手がトゲトゲダイヤモンド鉄球になった件について聞きたい。")
    rez = sc.query_with_rerank(
        "さのまるはダイヤモンド化して何をしてしまったの？ 起こったことを時系列に５つ教えて",
        lc.get_retriever(task_name)
    )
    print(rez)
    return rez

def rag_rerank_bm25(sc: Sirochatora):
    lc:LocalStorageRAG = LocalStorageRAG(
        "/mnt/d/dataset_for_ml/sanomaru",
        "txt"
    )
    task_name:str = "neko try"
    lc.fetch(task_name)
    lc.transform(task_name)
    lc.embed(task_name, "さのまるの手がトゲトゲダイヤモンド鉄球になった件について聞きたい。")
    rez = sc.query_with_rerank(
        "さのまるはダイヤモンド化して何をしてしまったの？ 起こったことを時系列に５つ教えて",
        lc.get_retriever(task_name, RetrievalType.BM25)
    )
    print(rez)
    return rez

def rag_rerank_hybrid(sc: Sirochatora):
    lc:LocalStorageRAG = LocalStorageRAG(
        "/mnt/d/dataset_for_ml/sanomaru",
        "txt", split_chunk_size=1000, split_chunk_overlap=150
    )
    task_name:str = "nekohybrid"
    lc.fetch(task_name)
    lc.transform(task_name)
    lc.embed(task_name, "さのまるの手がトゲトゲダイヤモンド鉄球になった件について聞きたい。")
    rez = sc.query_with_rerank_vm_sim_hybrid(
        "さのまるはダイヤモンド化して何をしてしまったの？ 起こったことを時系列に５つ教えて",
        lc.get_retriever(task_name, RetrievalType.CHROMA),
        lc.get_retriever(task_name, RetrievalType.BM25)
    )
    print(rez)
    return rez

def git_rag(sc:Sirochatora):
    task_name = "nekogit_langchain_sample"
    gitlag = GithubRAG(src = "https://github.com/langchain-ai/langchain", filter_cond = ".mdx")
    gitlag.set_branch_name("master")
    gitlag.fetch(task_name)

def ask_graph_qa(sc:Sirochatora):
    sc.graph_init_qaflow()
    print(sc.ask_with_graph("カモミールの効用について教えてください"))

def ask_graph_simple(sc:Sirochatora):
    sc.graph_init_simpletalk()
    print(sc.ask_with_graph("カモミールの効用について教えてください"))

def main():

    conf:ConfJsonLoader = ConfJsonLoader("sirochatora/conf.json")
    environ["LANGSMITH_API_KEY"] = conf._conf["LANGSMITH_API_KEY"]
    environ["LANGCHAIN_PROJECT"] = conf._conf["LANGCHAIN_PROJECT"]

    #sc:Sirochatora = Sirochatora(temperature=1.)
    #sc:Sirochatora = Sirochatora(temperature=0., is_chat_mode=True)
    sc:Sirochatora = Sirochatora(model_name = "hf.co/MaziyarPanahi/gemma-7b-GGUF:Q4_K_M", role_def_conf = "study_role.json")
    ask_graph_simple(sc)
    #ask_graph_qa(sc)
    #rag_rerank(sc)
    #rag_bm25(sc)
    #rag_rerank_bm25(sc)
    #rag_rerank_hybrid(sc)
    #git_rag(sc)
    #rag_multiqueries(sc)
    #rag_hyde(sc)
    #chat_sample(sc)
    #rez = rag_localdoc_q_with_ctx(sc)
    #rag_localdoc_simple()
    #ask_zero_cot(sc)
    #ask_negpos(sc)
    #rag_web(sc)
    
    #simple_template(sc)

if __name__ == "__main__":
    main()
