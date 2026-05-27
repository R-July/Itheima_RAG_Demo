import os
from time import strftime
import datetime
from sqlalchemy import false
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

"""
知识库
"""

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

if __name__ == "__main__":
    service = KnowledgeBaseService()
    r = service.upload_by_str("饶乐天","testfile")
    print(r)