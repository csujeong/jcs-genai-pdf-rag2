import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# Function to summarize text
def summarize_text(texts, docsearch, chain):
    summry = docsearch.similarity_search(" ")
    txt = chain.run(input_documents=summry, question="write summery in points within 150 words")
    return txt

# Function to answer question
def answer_question(query, docsearch, chain):
    docs = docsearch.similarity_search(query)
    txt = chain.run(input_documents=docs, question=query)
    return txt

# Main function
def main():
    st.title('Summarization and Questioning Model')

    api_key = st.text_input('Your Key', placeholder="Enter Your key")
    os.environ["OPENAI_API_KEY"] = api_key

    raw_text = ''

    # Upload PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

    temp_data = "data : "
    temp_data += st.text_area('Text to analyze', placeholder="Enter Your Data")

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    texts = text_splitter.split_text(raw_text)
    texts.append(temp_data)

    query = "query : "
    query += st.text_area('Query on Data', placeholder="Enter Your Query")

    col1, col2, col3 = st.columns(3)

    if col2.button('Submit') and (uploaded_file is not None or temp_data!="data : ") :
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        if 'summry' not in st.session_state or  query == "query : ":  # Check if summary is already computed
            # Call summarize_text() function
            st.session_state['summry'] = summarize_text(texts, docsearch, chain)
            st.write('Summary:', st.session_state['summry'])

        # Call answer_question() function
        if query != "query : ":
            txt = answer_question(query, docsearch, chain)
            st.write('Output:', txt)
    else:
        st.write('Enter Your Query !')

if __name__ == '__main__':
    main()
