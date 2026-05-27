from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from langchain_core.documents import Document
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, format_document, MessagesPlaceholder


class RagService(object):
    def __init__(self):

        self.vector_service = VectorStoreService(
            embeddings=DashScopeEmbeddings(model=config.embeddings_model_name)
        )

        self.prompt_template = ChatPromptTemplate(
            [
                ("system","以我提供的已知参考资料为主，"
                 "简洁和专业的回答用户问题。参考资料:{context}。"),
                ("system","并且我提供用户的对话历史，如下："),
                MessagesPlaceholder("history"),
                ("human","请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(model = config.chat_models_name)

        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()

        def format_document(docs:list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段:{doc.page_content}\n文档元数据:{doc.metadata}\n\n"

            return formatted_str

        def format_for_retriever(value:dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            # 这里要做到{input,context,history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        # 注意这里不能用传统的chain链，直接用 |
        '''
        刚开始要并行构造input和context
        {
            "input":RunnablePassthrough(),input和历史记录(是一个字典)进来后，这里原路返回，不做处理
            下面用来生成参考资料context，format_for_retriever用来只取用户问题，不要历史记录，放到检索器retriever里去检索，format_document负责把检索返回的消息转成一个字符串
            "context":RunnableLambda(format_for_retriever) | retriever | format_document
        } 
        上面的输出有些混乱，需要format_for_prompt_template来整理
        从下面整理出需要的东西：
        {
            "input": 
            {
                "input": "我身高170厘米，尺码推荐",
                "history": [...]
            },
            "context": "文档片段:身高170cm，体重60kg，建议选择M码。\n文档元数据:{...}\n\n"
        }
        变成：
        {
            "input": "我身高170厘米，尺码推荐",
            "context": "文档片段:身高170cm，体重60kg，建议选择M码。\n...",
            "history": [...]
        }
        后面就是常规链，给提示词，再给模型，最后用StrOutputParser()转为普通字符串用于输出
        '''
        chain = (
            {
                "input":RunnablePassthrough(),
                "context":RunnableLambda(format_for_retriever) | retriever | format_document
             } | RunnableLambda(format_for_prompt_template) | self.prompt_template | self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

if __name__ == "__main__":
    # session_id 配置
    session_config = {
        "configurable":{
            "session_id":"user_001"
        }
    }

    res = RagService().chain.invoke({"input":"我身高170厘米，尺码推荐"},session_config)
    print(res)