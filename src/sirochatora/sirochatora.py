from langchain_core.runnables import RunnablePassthrough, RunnableParallel, ConfigurableField
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI

from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import SecretStr, BaseModel, Field

from sirochatora.util.siroutil import from_system_message_to_tuple
from typing import Optional, Any, Annotated
from uuid import uuid4
from os import getenv, environ
import operator
import json

class QueryGenOutput(BaseModel):
    queries:list[str] = Field(...)

class Judgement(BaseModel):
    judge:bool = Field(default = False, description = "判定結果")
    reason:str = Field(default = "", description = "判定理由")

class MessageBasedState(BaseModel):
    query:str = Field(
        ..., description = "ユーザーからの質問"
    )
    current_role:str = Field(
        default = "", description = "選定された解答ロール"
    )
    messages:Annotated[list[BaseMessage], operator.add] = Field(default = [])
    validation_result:bool = Field(
        default = False, description = "品質チェックの結果"
    )
    validation_result_reason:str = Field(
        default = "", description = "品質チェックの判定理由"
    )

class State(BaseModel):
    query:str = Field(
        ..., description = "ユーザーからの質問"
    )
    current_role:str = Field(
        default = "", description = "選定された解答ロール"
    )
    messages:Annotated[list[str], operator.add] = Field(
        default = [], description = "回答履歴"
    )
    current_judge:bool = Field(
        default = False, description = "品質チェックの結果"
    )
    judgement_reason:str = Field(
        default = "", description = "品質チェックの判定理由"
    )

class Sirochatora:

    def __init__(self, 
                model_name:str = "gemma3:4b", 
                temperature:float = 0.1,
                is_chat_mode:bool = False,
                session_id:Optional[str] = None,
                role_def_conf:Optional[str] = None):
        self._model_name:str = model_name
        self._temperature = temperature
        self._system_msgs:list[SystemMessage] = [
            SystemMessage("Rule1: あなたはとても賢くてかわいいねこちゃんです。")
        ]
        self._llm = ChatOpenAI(
            model = self._model_name,
            api_key = SecretStr("ollama"),
            temperature = self._temperature,
            base_url = 'http://localhost:11434/v1'
        )
        self._llm = self._llm.configurable_fields(max_tokens = ConfigurableField(id = "max_tokens"))

        self._is_chat_mode = is_chat_mode
        if is_chat_mode:
            db_path = getenv("NEKOKAN_INDB_PATH")
            if session_id is None:
                session_id = uuid4().hex
            self._msg_history = SQLChatMessageHistory(
                session_id = session_id, 
                connection = f"sqlite:///{db_path}/sirochatora_chatv1.db")
        self._session_id = session_id
        self._role_def_conf = role_def_conf
        if role_def_conf is not None:
            conf_dir = environ["NEKORC_PATH"]
            with open(f"{conf_dir}/sirochatora/{role_def_conf}") as fp:
                self._roles = json.load(fp)
        self._is_graph_ready = False

    def graph_selection_node(self, state: State) -> dict[str, Any]:
        query = state.query
        role_options = "\n".join(f"{k}. {v['name']}: {v['description']}" for k, v in self._roles.items())
        prompt = ChatPromptTemplate.from_template(\
            """
            質問を分析して、最も適切な解答担当ロールを選択してください。

            選択肢:
            {role_options}

            回答は選択肢の番号(1,2または3)のみを返してください。

            質問: {query}
            """.strip()
        )
        chain = prompt | self._llm.with_config(configurable = dict(max_tokens = 1))|StrOutputParser()
        role_number = chain.invoke({"role_options": role_options, "query": query})

        selected_role = self._roles[role_number.strip()]["name"]
        return {"current_role": selected_role}

    def graph_answering_node(self, state: State) -> dict[str, Any]:
        query = state.query
        role = state.current_role
        role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in self._roles.values()])
        prompt = ChatPromptTemplate.from_template(
        """
        あなたは{role}として回答してください。以下の質問に対してあなたの役割に基づいた適切な回答を提供してください。
        役割の詳細: {role_details}

        質問: {query}

        回答:""".strip()
        )
        chain = prompt | self._llm| StrOutputParser()
        ans = chain.invoke({"role": role, "role_details": role_details, "query": query})
        return {"messages": [ans]}

    def graph_check_node(self, state:State) -> dict[str, Any]:
        query = state.query
        ans = state.messages[-1]

        prompt = ChatPromptTemplate.from_template(
        """
        以下の回答の品質をチェックし、問題がある場合は'False', 問題が無い場合は'True'を解答して下さい。
        また、その判断理由も説明してください。

        ユーザーからの質問: {query}
        回答: {answer}
        """.strip()
        )
        chain = prompt | self._llm.with_structured_output(Judgement) # type: ignore
        result:Judgement = chain.invoke({"query": query, "answer": ans}) # type: ignore

        return {
            "current_judge": result.judge,
            "judgement_reason": result.reason
        }

    def graph_set_system(self, _:MessageBasedState) -> dict[str, Any]:
        msg_buf = [SystemMessage(content = "Rule1: あなたはとても賢くてかわいいねこちゃんです。")]
        return {"messages": msg_buf}

    def graph_add_msg(self, state:MessageBasedState) -> dict[str, Any]:
        msg_buf = []
        msg_buf.append(HumanMessage(content = state.query))
        return {"messages": msg_buf}

    def graph_llm_resp(self, state:MessageBasedState) -> dict[str, Any]:
        ai_msg = self._llm.invoke(state.messages)
        return {"messages": [ai_msg]}

    def graph_init_simpletalk(self):
        g = StateGraph(MessageBasedState)
        g.add_node("set_system", self.graph_set_system)
        g.add_node("add_message", self.graph_add_msg)
        g.add_node("llm_response", self.graph_llm_resp)
        g.set_entry_point("set_system")
        g.add_edge("set_system", "add_message")
        g.add_edge("add_message", "llm_response")
        g.add_edge("llm_response", END)
        self._compiled_workflow = g.compile()
        self._is_graph_ready = True

    def graph_init_qaflow(self):
        workflow = StateGraph(State)
        workflow.add_node("selection", self.graph_selection_node)
        workflow.add_node("answering", self.graph_answering_node)
        workflow.add_node("check", self.graph_check_node)
        workflow.set_entry_point("selection")
        workflow.add_edge("selection", "answering")
        workflow.add_edge("answering", "check")
        workflow.add_conditional_edges(
            "check",
            lambda state: state.current_judge,
            {True: END, False: "selection"}
        )

        self._compiled_workflow = workflow.compile()
        self._is_graph_ready = True

    def ask_with_graph(self, q:str) -> str:
        if not self._is_graph_ready:
            return "<<GRAPH NOT READY>>"
        init_state = State(query = q)
        rez = self._compiled_workflow.invoke(init_state)
        print(rez)
        return rez["messages"][-1]

    def change_session_id(self, new_session_id:Optional[str] = None) -> None:
        if not self._is_chat_mode:
            raise RuntimeError("chat mode must be ON")
        if new_session_id is None:
            new_session_id = uuid4().hex
        self._session_id = new_session_id
        db_path = getenv("NEKOKAN_INDB_PATH")
        self._msg_history = SQLChatMessageHistory(
            session_id = self._session_id, 
            connection = f"sqlite:///{db_path}/sirochatora_chatv1.db")

    def clear_system(self) -> None:
        self._system_msgs.clear()

    def get_temperature(self) -> float:
        return self._temperature
    
    def append_system_message(self, sys_msg:str) -> None:
        if isinstance(self._system_msgs[-1].content, str):
            new_msg:str = self._system_msgs[-1].content + sys_msg
            self._system_msgs[-1] = SystemMessage(new_msg)

    def add_system_message(self, sys_msg:str) -> None:
        self._system_msgs.append(SystemMessage(sys_msg))

    def query(self, q:str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            from_system_message_to_tuple(self._system_msgs[-1]),
            ("human", q)
        ])
        chain = prompt | self._llm | StrOutputParser()
        if self._is_chat_mode:
            messages_from_hist = self._msg_history.get_messages()
            resp = chain.invoke({
                "chat_history": messages_from_hist,
                "question": q
            })
            self._msg_history.add_user_message(q)
            self._msg_history.add_ai_message(resp)
            print(f"@query:session_id - {self._session_id}")
            return resp
        else:
            return chain.invoke({
                "question": q
            })
        

    def query_with_template(self, q:str, vals:dict[str, str]) -> str:
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                from_system_message_to_tuple(self._system_msgs[-1]),
                ("human", q)
            ]
        )
        chain = prompt_template | self._llm | StrOutputParser()
        return chain.invoke(vals)

    def query_within_context(self, q:str, retriever: BaseRetriever) -> str:

        prompt = ChatPromptTemplate.from_template('''\
            以下の文脈だけを踏まえて質問に解答してください。

            文脈:"""
            {context}
            """

            質問:"""
            {question}
            """
        ''')

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self._llm
            | StrOutputParser()
        )
        return chain.invoke(q)

    def query_with_retriever(self, q:str, retriever: BaseRetriever) -> str:

        prompt = ChatPromptTemplate.from_template('''\
            以下の文脈だけを踏まえて質問に解答してください。

            文脈:"""
            {context}
            """

            質問:"""
            {question}
            """
        ''')

        chain = {
            "question": RunnablePassthrough(),
            "context": retriever
        } | prompt | self._llm | StrOutputParser()

        return chain.invoke(q)


    # HYpothetical Document Embedding
    def query_with_hyde(self, q:str, retriever: BaseRetriever) -> str:

        str_parser = StrOutputParser()
        hyde_prompt = ChatPromptTemplate.from_template("""\
            次の質問に解答する一文を書いてください。

            質問: {question}
        """)
        hyde_chain = hyde_prompt | self._llm | str_parser

        prompt = ChatPromptTemplate.from_template('''\
            以下の文脈だけを踏まえて質問に解答してください。

            文脈:"""
            {context}
            """

            質問:"""
            {question}
            """
        ''')

        hyde_rag_chain = {
            "question": RunnablePassthrough(),
            "context": hyde_chain | retriever
        } | prompt | self._llm | str_parser

        return hyde_rag_chain.invoke(q)

    def reciprocal_rank_fusion(self, 
                            retriever_outs: list[list[Document]],
                            k:int = 60,
                            top_n:int = 3)  -> list[str]:
        content_scores = {}
        for docs in retriever_outs:
            for rank, doc in enumerate(docs):
                content = doc.page_content

                if content not in content_scores:
                    content_scores[content] = 0
                content_scores[content] += 1 / (rank + k)
        
        rank_sorted = sorted(content_scores.items(), key = lambda x:x[1], reverse = True)
        if top_n != -1:
            return [content for content, _ in rank_sorted][:top_n]
        else:
            return [content for content, _ in rank_sorted]


    def _rerank(self, d: dict[str, Any], top_n:int = 3) -> list[Document]:
        q = d["question"]
        documents = d["documents"]

        compressor = CohereRerank(model = "rerank-multilingual-v3.0", top_n = top_n)
        return list(compressor.compress_documents(documents = documents, query = q))

    def query_with_rerank(self, q:str, retriever:BaseRetriever) -> str:

        prompt = ChatPromptTemplate.from_template('''\
            以下の文脈だけを踏まえて質問に解答してください。

            文脈:"""
            {context}
            """

            質問:"""
            {question}
            """
        ''')

        rerank_chain = {
            "question": RunnablePassthrough(),
            "documents": retriever
        } | RunnablePassthrough.assign(context = self._rerank) | prompt | self._llm | StrOutputParser()

        return rerank_chain.invoke(q)

    def query_with_rerank_vm_sim_hybrid(
            self,
            q:str,
            v_retriver:BaseRetriever,
            bm25_retriver:BaseRetriever
    ) -> str:

        prompt = ChatPromptTemplate.from_template('''\
            以下の文脈だけを踏まえて質問に解答してください。

            文脈:"""
            {context}
            """

            質問:"""
            {question}
            """
        ''')

        hybrid_chain = (
            RunnableParallel({
                "sim_docs": v_retriver,
                "bm25_docs": bm25_retriver
            }) | (lambda x:[x["sim_docs"], x["bm25_docs"]]) | self.reciprocal_rank_fusion
        )
        
        rag_chain = (
            {
                "question": RunnablePassthrough(),
                "context": hybrid_chain
            } | prompt | self._llm | StrOutputParser()
        )

        return rag_chain.invoke(q)


    def query_with_multiqueries(self, q:str, retriever:BaseRetriever) -> str:
        multi_query_gen_prompt = ChatPromptTemplate.from_template("""\
        質問に対してベクターデータベースから関連文書を検索するため３つの異なる検索クエリを生成してください。
        距離ベースの類似性検索の限界を克服するためユーザーの質問に対して複数の視点を提供するのが目的です。

        質問: {question}
        """)

        multi_query_gen_chain = (multi_query_gen_prompt 
        | self._llm.with_structured_output(QueryGenOutput)
        | (lambda x: x.queries))

        prompt = ChatPromptTemplate.from_template('''\
            以下の文脈だけを踏まえて質問に解答してください。

            文脈:"""
            {context}
            """

            質問:"""
            {question}
            """
        ''')

        multi_query_rag_chain = {
            "question": RunnablePassthrough(),
            "context": multi_query_gen_chain | retriever.map() | self.reciprocal_rank_fusion
        } | prompt | self._llm | StrOutputParser()

        return multi_query_rag_chain.invoke(q)


    def query_with_zeroshot_CoT(self, q:str, override_system:bool = True) -> str:

        out_parser = StrOutputParser()
        if override_system:
            system = ("system", "ユーザーの質問にステップバイステップで答えてください。")
        else:
            system = from_system_message_to_tuple(self._system_msgs[-1])
            system = (system[0], f"{system[1]}; ユーザーの質問にステップバイステップで答えてください。")

        zero_prompt = ChatPromptTemplate.from_messages([system, ("human", "{q}")])
        zero_chain = zero_prompt | self._llm | out_parser

        if override_system:
            system = ("system", "ステップバイステップで考えた回答から結論を抽出してください")
        else:
            system = from_system_message_to_tuple(self._system_msgs[-1])
            system = (system[0], f"{system[1]}; ステップバイステップで考えた回答から結論を抽出してください")

        cot_prompt = ChatPromptTemplate.from_messages([system, ("human", "{text}")])
        cot_chain = cot_prompt | self._llm | out_parser

        cot_conclusion_chain = zero_chain | cot_chain
        return cot_conclusion_chain.invoke({"q": q})

    def query_with_negpos_resp(self, q:str, override_system: bool = True) -> str:

        out_parser = StrOutputParser()

        if override_system:
            system = ("system", "楽観主義的で明るく快活な意見を述べてください。")
        else:
            system = from_system_message_to_tuple(self._system_msgs[-1])
            system = (system[0], f"{system[1]}; 楽観主義的で明るく快活な意見を述べてください。")

        opt_prompt = ChatPromptTemplate.from_messages([system, ("human", "{topic}")])
        opt_chain = opt_prompt | self._llm | out_parser

        if override_system:
            system = ("system", "悲観主義的で暗く皮肉的な意見を述べてください。")
        else:
            system = from_system_message_to_tuple(self._system_msgs[-1])
            system = (system[0], f"{system[1]}; 悲観主義的で暗く皮肉的な意見を述べてください。")

        pesi_prompt = ChatPromptTemplate.from_messages([system, ("human", "{topic}")])
        pesi_chain = pesi_prompt | self._llm | out_parser

        if override_system:
            system = ("system", "客観的に２つの意見をまとめてください。")
        else:
            system = from_system_message_to_tuple(self._system_msgs[-1])
            system = (system[0], f"{system[1]}; ２つの意見をまとめてください。")

        synth_prompt = ChatPromptTemplate.from_messages(
            [
                system,
                ("human", "楽観的意見: {optimistic_opinion}\n 悲観的意見: {pesimistic_opinion}")
            ]
        )

        synth_chain = RunnableParallel(
            { 
                "optimistic_opinion": opt_chain,
                "pesimistic_opinion": pesi_chain
            }
        ) | synth_prompt | self._llm | out_parser

        return synth_chain.invoke({"topic": q})
