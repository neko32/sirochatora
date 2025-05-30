from libs.sirochatora.sirochatora import Sirochatora
from libs.sirochatora.rag.rag import LocalStorageRAG, TavilyRAG
from libs.sirochatora.util.siroutil import ConfJsonLoader, ctxdict_to_str
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

def main():

    conf:ConfJsonLoader = ConfJsonLoader("sirochatora/conf.json")
    environ["LANGSMITH_API_KEY"] = conf._conf["LANGSMITH_API_KEY"]
    environ["LANGCHAIN_PROJECT"] = conf._conf["LANGCHAIN_PROJECT"]

    #sc:Sirochatora = Sirochatora(temperature=1.)
    sc:Sirochatora = Sirochatora(temperature=0.3, is_chat_mode=True)
    chat_sample(sc)
    #rez = rag_localdoc_q_with_ctx(sc)
    #rag_localdoc_simple()
    #ask_zero_cot(sc)
    #ask_negpos(sc)
    #rag_web(sc)
    
    #simple_template(sc)

if __name__ == "__main__":
    main()
