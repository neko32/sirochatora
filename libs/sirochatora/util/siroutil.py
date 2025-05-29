from langchain_core.messages import SystemMessage, HumanMessage
import json
from os import getenv

class ConfJsonLoader:
    def __init__(self, loc:str):
        nekorc = getenv("NEKORC_PATH")
        if nekorc is None:
            raise RuntimeError(f"{loc} not found")
        with open(f"{nekorc}/{loc}", "r") as fp:
            js = json.load(fp)
            self._conf = js

def from_system_message_to_tuple(sys_msg:SystemMessage) -> tuple[str, str]:
    if isinstance(sys_msg.content, str):
        return (
            "system",
            sys_msg.content
        )
    else:
        raise RuntimeError("SystemMessage Content is not str type")
