import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    print(os.environ["PINECONE_API_KEY"])

    loader = UnstructuredLoader(file_path="/Users/arunvijapurapu/Desktop/Python/github/langchain-course-rag/mediumblog1.txt", chunking_strategy="basic", max_characters=1000000)
    documents = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"created {len(texts)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.environ["GOOGLE_API_KEY"]
    , model="gemini-embedding-2")
    
    print("ingesting...")
  
    PineconeVectorStore.from_documents(
    texts, embeddings, index_name=os.environ["INDEX_NAME"])

    print("finish")
    