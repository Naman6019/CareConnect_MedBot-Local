import os

# Ensure only one OpenMP runtime is used
os.environ["KMP_INIT_AT_FORK"] = "FALSE"  # Prevent issues with subprocesses
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow execution but risks remain

import torch
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Limit PyTorch threads
torch.set_num_threads(1)

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/faiss_db.index'  # This is your directory path

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': device})

    all_embeddings = [embeddings.embed_query(doc.page_content) for doc in texts]
    all_embeddings = np.array(all_embeddings).astype('float32')

    dim = all_embeddings.shape[1]  # Get the dimension of your embeddings
    index = faiss.IndexFlatL2(dim)  # L2 distance metric (Euclidean distance)
    index.add(all_embeddings)  # Add embeddings to the index
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    
    # Save the index to 'index.faiss' file explicitly
    faiss.write_index(index, os.path.join(os.path.dirname(DB_FAISS_PATH), 'index.faiss'))
    
    print(f"FAISS index created and saved at {os.path.join(os.path.dirname(DB_FAISS_PATH), 'index.faiss')}")

if __name__ == "__main__":
    create_vector_db()
