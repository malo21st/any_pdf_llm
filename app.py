import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from tempfile import NamedTemporaryFile
from PIL import Image
import os
import locale

locale.setlocale(locale.LC_ALL, "ja_JP.UTF-8")

os.environ["OPENAI_API_KEY"] = st.secrets.openai_api_key

INTRO = "この文章を３０字程度で要約して下さい。　回答後は、必ず'改行'して「ご質問をどうぞ。」を付けて下さい。"

if "qa" not in st.session_state:
    st.session_state.qa = {"pdf": "", "history": []}
#     st.session_state["qa"] = {"pdf": "", "history": [{"role": "Q", "msg": INTRO}]}

# Prompt
template = """
質問に日本語で回答してください。
# 質問：{question}
# 回答 IN JAPANESE：
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
def get_vector_db(uploaded_file):
    with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
        f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(f.name)
        documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(texts, embeddings)

def store_del_msg():
    st.session_state.qa["history"].append({"role": "Q", "msg": st.session_state.user_input}) # store
    st.session_state.user_input = ""  # del

# View (User Interface)
## Sidebar
st.sidebar.title("ＰＤＦアシスタント")
uploaded_file = st.sidebar.file_uploader("PDFファイルをアップロードして下さい", type=["pdf"])
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.qa.get("pdf", ""):
        st.session_state.qa = {"pdf": uploaded_file.name, "history": []}
    user_input = st.sidebar.text_input("ご質問をどうぞ", key="user_input", on_change=store_del_msg)
#     st.sidebar.markdown("---")
#     st.sidebar.write(uploaded_file.name)
    ## Main Content
    if st.session_state.qa["history"]:
        for message in st.session_state.qa["history"]:
#         for message in st.session_state["qa"][1:]:
            if message["role"] == "Q": # Q: Question (User)
                st.info(message["msg"])
            elif message["role"] == "A": # A: Answer (AI Assistant)
                st.success(message["msg"])
            elif message["role"] == "E": # E: Error
                st.error(message["msg"])
    chat_box = st.empty() # Streaming message

    # Model (Business Logic)
    vectordb = get_vector_db(uploaded_file)
    stream_handler = StreamHandler(chat_box)
    chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, callbacks=[stream_handler], temperature=0)
    qa = RetrievalQA.from_chain_type(llm=chat_llm, chain_type="stuff", retriever=vectordb.as_retriever())
    if st.session_state.qa["history"]:
        query = "・" + st.session_state.qa["history"][-1]["msg"]
        try:
            response = qa.run(query) # Query to ChatGPT
            st.session_state.qa["history"].append({"role": "A", "msg": response})
        except Exception:
            response = "エラーが発生しました！　もう一度、質問して下さい。"
            st.error(response)
            st.session_state.qa.qa["history"].append({"role": "E", "msg": response})
