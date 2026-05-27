"""
基于Streamlit完成web网页上传服务

注意：当Web页面元素发生变化的时候，代码都是重新运行了一次，状态无法自动保持
"""
import time
import streamlit as st
from knowledge_base import KnowledgeBaseService


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


