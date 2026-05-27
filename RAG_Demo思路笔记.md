# 整体流程：

![整体流程图](C:\Users\13114\Desktop\学习文件\Agent\黑马RAG_Demo\整体流程图.png)

## 离线流程：

![离线_知识库搭建_流程图](C:\Users\13114\Desktop\学习文件\Agent\黑马RAG_Demo\离线_知识库搭建_流程图.png)

## 在线流程：

[![在线_检索_流程图](C:\Users\13114\Desktop\学习文件\Agent\黑马RAG_Demo\在线_检索_流程图.png)]()

#  app_file_upload

用于上传资料，还只是把资料传上去，还没有存入知识库

```python
# 添加标题
st.title("知识库更新")
# file_uploader
uploader_file = st.file_uploader(
    "请上传文件",
    type=["txt"],
    accept_multiple_files=True,    # False表示只接受一个文件的上传
)

# session_state是一个字典，里面的状态可以保持,不然Streamlit每次运行都是从头开始刷新程序
# 这样KnowledgeBaseService类对象只要创建一次，不会每次都重新创建
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    #   提取文件内的信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size/1024 #KB

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type}|大小:{file_size:.2f}KB")

    # 获取上传文件中的内容
    # get_value->bytes->decode('utf-8')，解码成字符串
    text = uploader_file.getvalue().decode("utf-8")

    with st.spinner("载入知识库中......"):
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text,file_name)
        st.write(result)
```

# knowledge_base

把资料转存入知识库Chroma里，这里还有清洗，切片等操作

```python
# (md5_str:str)表示传进来的md5_str是str类型
def check_md5(md5_str:str):
    """检查传入的md5字符串是否已经被处理过"""
    if not os.path.exists(config.md5_path):
        # if进入的文件不存在，表示没有处理过这个md5
        open(config.md5_path,'w',encoding='utf-8').close()
        return False        # 表示还没处理
    else:
        # 逐行打开
        for line in open(config.md5_path,'r',encoding='utf-8').readlines():
            line = line.strip()     # 处理字符串前后的回车和空格
            if line == md5_str:
                return True     # 表示已经处理过了

        return False        # for循环跑完都没True，说明是False

def save_md5(md5_str:str):
    """将传入的md5字符串，记录到文件内保存"""
    # 这里with用于资源管理，比如用完md5会自动关闭
    with open(config.md5_path,'a',encoding='utf-8') as f:
        f.write(md5_str + '\n')

def get_string_md5(input_str:str,encoding='utf-8'):
    """将传入的字符串转为md5字符串"""

    # 把字符串转为bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)
    # 创建md5对象
    md5_boj = hashlib.md5()         # 得到md5对象
    md5_boj.update(str_bytes)       # 更新内容
    md5_hex = md5_boj.hexdigest()   # 得到md5的十六进制字符串

    return md5_hex


class KnowledgeBaseService(object):
    def __init__(self):
        # 如果文件夹不存在则创建，存在就跳过
        os.makedirs(config.persist_directory,exist_ok=True)

        # 向量存储的实例，Chroma向量库对象
        self.chroma = Chroma(
            collection_name=config.collection_name,     # 数据库的表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory,  # 数据库本地储存地址
        )
        
        # 文本分割器的对象吗，chunk分割
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,       # 分割后的文本段最大长度
            chunk_overlap=config.chunk_overlap, # 连续文本之间的重叠数量
            separators=config.separators,      # 自然段落划分的符号
            length_function=len,                # 使用Python自带的len函数作长度统计
        )


    def upload_by_str(self,data,filename):
        """将传入的字符串，向量化，存入向量数据库中"""
        # 先得到传入字符串的md5值
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"
        # 太长就分割
        if len(data) > config.max_spliter_char_number:
            knowledge_chunks:list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        # 元数据metadata自定义
        metadata = {
            "scoure":filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"小饶"
        }

        self.chroma.add_texts(  # 内容就加载到向量库中了
            knowledge_chunks,
            metadatas = [metadata for _ in knowledge_chunks],    # 因为上面传进来knowledge_chunks—>list[str],所以这里for循环对应格式
        )

        save_md5(md5_hex)

        return "[成功]内容已经成功载入向量库"
```

# vector_stores

定义检索器，针对用户的提问，去知识库里检索匹配

```python
class VectorStoreService(object):
    def __init__(self,embeddings):
        # 传入嵌入模型
        self.embeddings = embeddings

        self.vector_store = Chroma(
            collection_name = config.collection_name,
            embedding_function = self.embeddings,
            persist_directory = config.persist_directory,
        )

    def get_retriever(self):
        """返回向量检索器，方便加入chain"""
        return self.vector_store.as_retriever(search_kwargs={"k": config.simple_threshold})
```

# rag

最重要的chain链构建，还有历史增强

```python
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
```

# file_history_store

历史记录增强模块，用作对话记录保存

```python
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
```

# app_qa

最后交互页面

```python
st.title("柒月的智能客服")
st.divider()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant","content":"你好，有什么可以帮助你？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

# 在页面最下面提供用户输入栏
prompt = st.chat_input()

if prompt:
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role":"user","content":prompt})

    with st.spinner("AI思考中......"):
        time.sleep(1)
        res = st.session_state["rag"].chain.invoke({"input":prompt},config.session_config)
        st.chat_message("assistant").write(res)
        st.session_state["messages"].append({"role": "assistant", "content": res})
```

# 文件等级

![image-20260513095837810](C:\Users\13114\AppData\Roaming\Typora\typora-user-images\image-20260513095837810.png)

# config定义

```python
md5_path = "./md5.text"

# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"

# spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n","\n",".","!","?","。","！","？"," ",""]
max_spliter_char_number = 1000      # 文本分割的阈值

#
simple_threshold = 1        # 检索返回匹配的文档数量

embeddings_model_name = "text-embedding-v4"
chat_models_name = "qwen3-max"

#
session_config = {
    "configurable": {
        "session_id": "user_001"
    }
}
```