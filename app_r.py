import streamlit as st
from PyPDF2 import PdfReader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from PIL import Image
import os

os.environ["OPENAI_API_KEY"] = st.secrets.openai_api_key

INTRO = "３０文字程度で要約して下さい。　回答後は、必ず'改行'して「ご質問をどうぞ。」を付けて下さい。"
if "qa" not in st.session_state:
    st.session_state["qa"] = []
#     st.session_state["qa"] = [{"role": "Q", "msg": INTRO}]

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
def get_vector_db(pdf):
    pdf_reader = PdfReader(pdf)
    documents = ""
    for page in pdf_reader.pages:
        documents += page.extract_text()
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(documents)
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    return Chroma.from_texts(texts, embeddings)

def store_del_msg():
    st.session_state["qa"].append({"role": "Q", "msg": st.session_state["user_input"]}) # store
    st.session_state["user_input"] = ""  # del

# View (User Interface)
## Sidebar
st.sidebar.title("ＰＤＦアシスタント")
uploaded_file = st.sidebar.file_uploader("PDFファイルをアップロードして下さい", type=["pdf"])
if uploaded_file is not None:
    user_input = st.sidebar.text_input("ご質問をどうぞ", key = "user_input", on_change = store_del_msg)
#     st.sidebar.markdown("---")
#     st.sidebar.write(uploaded_file.name)
    ## Main Content
    if st.session_state["qa"]:
        for message in st.session_state["qa"]:
#         for message in st.session_state["qa"][1:]:
            if message["role"] == "Q": # Q: Question (User)
                st.info(message["msg"])
            elif message["role"] == "A": # A: Answer (AI Assistant)
                st.success(message["msg"])
            elif message["role"] == "E": # E: Error
                st.error(message["msg"])
    chat_box=st.empty() # Streaming message

    # Model (Business Logic)
    vectordb = get_vector_db(uploaded_file)
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
