import os
import chainlit as cl
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import torch

# Absolute path to the FAISS index
DB_FAISS_PATH = os.path.join(os.getcwd(), 'vectorstore')

# Custom prompt for QA
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Function to set custom prompt
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Load Language Model
def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=256,
        temperature=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return llm

# QA Bot function
def qa_bot():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': device})

    # Load FAISS index from the local path
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise FileNotFoundError(f"FAISS database not found at {DB_FAISS_PATH}! Error: {str(e)}")

    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': qa_prompt}
    )
    return qa

# Chainlit handlers
@cl.on_chat_start
async def start():
    try:
        # Initialize the QA bot and store it in the user session
        chain = qa_bot()
        cl.user_session.set("chain", chain)
        await cl.Message(content="Hi, Welcome to CareConnect. What is your query?").send()
    except Exception as e:
        # Send error message if QA bot fails to load
        await cl.Message(content=f"Error loading the QA bot: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    conversation_context = message.content

    try:
        # Generate response using the LLM
        result = chain({"query": conversation_context})
        answer = result.get("result", "Sorry, I couldn't find an answer.")
        sources = result.get("source_documents", [])

        # Format the sources
        formatted_sources = ""
        if sources:
            formatted_sources = "\n\n**Sources:**\n"
            for doc in sources:
                source_name = doc.metadata.get('source', 'Unknown Source')
                page_number = doc.metadata.get('page', 'N/A')
                formatted_sources += f"- {source_name} (Page {page_number})\n"
        else:
            formatted_sources = "\nNo sources found."

        # Combine the answer with formatted sources
        final_output = f"{answer}{formatted_sources}"

        # Send the response to the user
        await cl.Message(content=final_output).send()
    except Exception as e:
        # Send error message if something goes wrong during the query handling
        await cl.Message(content=f"An error occurred: {str(e)}").send()
