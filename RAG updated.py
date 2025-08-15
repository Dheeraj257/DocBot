import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader, CSVLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from huggingface_hub import InferenceClient
import tempfile


load_dotenv()
hf_api_key = os.getenv("hugging_face_hub_api_token")


st.set_page_config(page_title=" DocBot", layout="wide")
st.title(" Ask the Doc")


def load_document(file):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())

    ext = file.name.lower().split(".")[-1]

    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path)
    elif ext in ["doc", "docx"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext in ["md", "markdown"]:
        loader = UnstructuredMarkdownLoader(file_path)
    elif ext == "csv":
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()


uploaded_file = st.file_uploader("Upload your Document", type=["pdf", "txt", "docx", "md", "csv"])

if uploaded_file:
    with st.spinner("Processing document..."):
        documents = load_document(uploaded_file)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma.from_documents(chunks, embedding_model)  # No persistence

        st.success("PDF processed successfully! You can now ask questions.")

        question = st.text_input("Ask a question about your PDF")
        if st.button("Get Answer") and question:
            retrieved_docs = db.similarity_search(question, k=3)
            reference_data = "\n\n".join([doc.page_content for doc in retrieved_docs])

            client = InferenceClient(provider="cerebras", api_key=hf_api_key)
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Keep responses concise."},
                    {"role": "user", "content": f"Answer this question: {question}\n\nReference:\n{reference_data}"}
                ],
            )

            st.markdown("**Answer:**")
            st.write(completion.choices[0].message["content"])
