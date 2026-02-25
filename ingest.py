
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

os.environ["OPENAI_API_KEY"] = "dummy-not-needed-for-deployment"

def ingest():
    loader = PyPDFLoader("resume.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Delete old chroma_db to start fresh
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    print(f"✅ Indexed {len(chunks)} chunks from your resume.")

if __name__ == "__main__":
    ingest()