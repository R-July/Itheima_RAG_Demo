from chromadb.rate_limit import simple_rate_limit
from langchain_ollama import chat_models
from pydantic.v1.validators import max_str_intmd5_path = "./md5.text"

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

