from langchain_core.chat_history import BaseChatMessageHistory
import os,json
from typing import Sequence
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

def get_history(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):
    # __init__表示构造类函数
    def __init__(self,session_id,storage_path):
        self.session_id = session_id        # 会话id
        self.storage_path = storage_path    #不同的id的存储文件，所在的文件夹路径
        # 完整的文件夹路径
        self.file_path = os.path.join(self.storage_path,self.session_id)
        # 确保文件夹存在
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

    # def 函数名(self, 参数名: 参数类型) -> 返回值类型:
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages)  # 已有的消息列表,是用 @property 定义成属性的
        all_messages.extend(messages)       # 新的和已有的融合成一个list
        # 把数据同步写入到本地文件中
        # 类对象写入文件->一堆二进制
        # 为了方便看，把消息转为字典，借助json模块
        new_messages = []
        for message in all_messages:
            d = message_to_dict(message)
            new_messages.append(d)
        # 将数据写入文本

        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_messages,f)

    @property   # @property装饰器把message 方法变成成员属性用
    def messages(self) -> list[BaseMessage]:
        # 当前文件内：list[字典]
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messages = json.load(f) # 返回值是：list[字典]
                return messages_from_dict(messages)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)