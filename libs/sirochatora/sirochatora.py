from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever

from langchain_openai import ChatOpenAI

from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import SecretStr

from libs.sirochatora.util.siroutil import from_system_message_to_tuple

class Sirochatora:

    def __init__(self, model_name:str = "gemma3:4b", temperature:float = 0.1):
        self._model_name:str = model_name
        self._temperature = temperature
        self._system_msgs:list[SystemMessage] = [
            SystemMessage("Rule1: You are a very smart cat.")
        ]
        self._llm = ChatOpenAI(
            model = self._model_name,
            api_key = SecretStr("ollama"),
            temperature = self._temperature,
            base_url = 'http://localhost:11434/v1'
        )


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
        messages = self._system_msgs + [HumanMessage(q)]
        ai_resp = self._llm.invoke(messages)
        content = ai_resp.content
        if isinstance(content, str):
            return content
        else:
            raise RuntimeError("return value must be str type here")

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
