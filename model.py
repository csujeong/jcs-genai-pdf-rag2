from io import StringIO
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

st.title('Summerisation Model')

api_key = st.text_input('Your Key',placeholder="Enter Your key")
os.environ["OPENAI_API_KEY"] = api_key

temp = st.slider('Temprature', 0.1, 1.0, 0.1)
st.write("Temprature ", temp)

length = st.slider('Max Characters', 200, 4000, 200)
st.write("Characters ", length)

raw_text = ''

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    # bytes_data = uploaded_file.getvalue()
    stringio = StringIO(uploaded_file.getvalue().decode("ISO-8859-1"))
    st.write(stringio)
    string_data = stringio.read()
    raw_text+=string_data
    # reader = PdfReader(bytes_data)
    # for i, page in enumerate(reader.pages):
    #     text = page.extract_text()
    #     if text:
    #         raw_text += text

temp_data = st.text_area('Text to analyze',placeholder="Eneter Your Data")

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)

texts = text_splitter.split_text(raw_text)
texts.append(temp_data)

query = st.text_area('Query on Data',placeholder="Enter Your Query")

if st.button('Submit'):
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(max_tokens=int(length),model_name="gpt-3.5-turbo",temperature=float(temp)), chain_type="stuff")

    docs = docsearch.similarity_search(query)  
    txt=chain.run(input_documents=texts, question="Summerize within 150 words")
    st.write('Summery :', txt)
    txt=chain.run(input_documents=docs, question=query)
    st.write('Output :', txt)
else:
    st.write('Enter Your Query !')