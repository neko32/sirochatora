from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.chat_message_histories import SQLChatMessageHistory

from langchain_openai import ChatOpenAI

from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import SecretStr, BaseModel, Field

from libs.sirochatora.util.siroutil import from_system_message_to_tuple
from typing import Optional
from uuid import uuid4
from os import getenv

class QueryGenOutput(BaseModel):
    queries:list[str] = Field(...)

class Sirochatora:

    def __init__(self, 
                model_name:str = "gemma3:4b", 
                temperature:float = 0.1,
                is_chat_mode:bool = False,
                session_id:Optional[str] = None):
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
        self._is_chat_mode = is_chat_mode
        if is_chat_mode:
            db_path = getenv("NEKOKAN_INDB_PATH")
            if session_id is None:
                session_id = uuid4().hex
            self._msg_history = SQLChatMessageHistory(
                session_id = session_id, 
                connection = f"sqlite:///{db_path}/sirochatora_chatv1.db")
        self._session_id = session_id

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

    def query_within_context(self, q:str, retriever: VectorStoreRetriever) -> str:

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

    # HYpothetical Document Embedding
    def query_with_hyde(self, q:str, retriever: VectorStoreRetriever) -> str:

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

    def query_with_multiqueries(self, q:str, retriever:VectorStoreRetriever) -> str:
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
            "context": multi_query_gen_chain | retriever.map()
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
