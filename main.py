from libs.sirochatora.sirochatora import Sirochatora
from libs.sirochatora.rag.rag import LocalStorageRAG
from libs.sirochatora.util.siroutil import ConfJsonLoader
from os import environ


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
    rez = lc.query_with_rag(task_name, "さのまるとトゲトゲダイヤモンド鉄球について聞きたい。")
    print(rez)
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

def main():

    conf:ConfJsonLoader = ConfJsonLoader("sirochatora/conf.json")
    environ["LANGSMITH_API_KEY"] = conf._conf["LANGSMITH_API_KEY"]
    environ["LANGCHAIN_PROJECT"] = conf._conf["LANGCHAIN_PROJECT"]

    #sc:Sirochatora = Sirochatora(temperature=1.)
    sc:Sirochatora = Sirochatora(temperature=0)
    rez = rag_localdoc_q_with_ctx(sc)
    
    #simple_template(sc)

if __name__ == "__main__":
    main()
