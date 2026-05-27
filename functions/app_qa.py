import time
from rag import RagService
import streamlit as st
from langchain import messages
import config_data as config

# 启动环境命令
# & "E:\Users\13114\anaconda3\envs\langchain_rag_learning\python.exe" -m streamlit run "C:\Users\13114\Desktop\学习文件\Agent\黑马RAG_Demo\app_qa.py"
# & "E:\Users\13114\anaconda3\envs\langchain_rag_learning\python.exe" -m streamlit run "C:\Users\13114\Desktop\学习文件\Agent\黑马RAG_Demo\app_file_upload.py"

st.title("柒月的智能客服")
st.divider()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant","content":"你好，有什么可以帮助你？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

# 显示历史聊天记录
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

# 在页面最下面提供用户输入栏
prompt = st.chat_input()

if prompt:
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    # 存进记录里
    st.session_state["messages"].append({"role":"user","content":prompt})

    with st.spinner("AI思考中......"):
        time.sleep(1)
        res = st.session_state["rag"].chain.invoke({"input":prompt},config.session_config)
        st.chat_message("assistant").write(res)
        st.session_state["messages"].append({"role": "assistant", "content": res})