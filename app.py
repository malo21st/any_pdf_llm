import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from PIL import Image
import os

os.environ["OPENAI_API_KEY"] = st.secrets.openai_api_key

INTRO = "この文書の概要を教えてください。　回答後は、必ず'改行'して「質問をどうぞ。」を付けて下さい。"
if "qa" not in st.session_state:
    st.session_state["qa"] = [{"role": "Q", "msg": INTRO}]

# Prompt
template = """
質問に日本語で回答してください。
# 質問：{question}
# 回答：
"""

prompt = PromptTemplate(
    input_variables = ["question"],
    template = template,
)

# Class and Function
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text = ""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text+=token 
        self.container.success(self.text) 

@st.cache_resource
def load_vector_db():
    return Chroma(persist_directory = "VECTOR_DB", embedding_function = OpenAIEmbeddings())

@st.cache_data
def load_image():
    return Image.open('tenjikai.png')

def store_del_msg():
    st.session_state["qa"].append({"role": "Q", "msg": st.session_state["user_input"]}) # store
    st.session_state["user_input"] = ""  # del

# Load data
vectordb = load_vector_db()
image = load_image() 

# View (User Interface)
## Sidebar
st.sidebar.title("補助金さん")
st.sidebar.write("補助金・助成金についてお任せあれ")
user_input = st.sidebar.text_input("ご質問をどうぞ", key = "user_input", on_change = store_del_msg)
st.sidebar.markdown("---")
st.sidebar.image(image, caption = '展示会出展助成事業（令和５年度　東京都）', use_column_width = "auto")
## Main Content
if st.session_state["qa"]:
    for message in st.session_state["qa"][1:]:
        if message["role"] == "Q": # Q: Question (User)
            st.info(message["msg"])
        elif message["role"] == "A": # A: Answer (AI Assistant)
            st.success(message["msg"])
        elif message["role"] == "E": # E: Error
            st.error(message["msg"])
chat_box=st.empty() # Streaming message

# Model (Business Logic)
stream_handler = StreamHandler(chat_box)
chat_llm = ChatOpenAI(model_name = "gpt-3.5-turbo", streaming = True, callbacks = [stream_handler])
qa = RetrievalQA.from_chain_type(llm = chat_llm, chain_type = "stuff", retriever = vectordb.as_retriever())
if st.session_state["qa"]: 
    query = "・" + st.session_state["qa"][-1]["msg"]
    try:
        response = qa.run(query) # Query to ChatGPT
        st.session_state["qa"].append({"role": "A", "msg": response})
    except Exception:
        response = "エラーが発生しました！　もう一度、質問して下さい。"
        st.error(response)
        st.session_state["qa"].append({"role": "E", "msg": response})
