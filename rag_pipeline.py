import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    raise EnvironmentError("GROQ_API_KEY environment variable not set (needed for the LLM).")

def create_vector_store(file_path, persist_directory):
    """Load a document, split it, embed it, and save to Chroma."""
    print(f"Starting to process file: {file_path}")

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF and TXT are supported.")

    documents = loader.load()
    if not documents:
        print("No documents loaded. Check the file content.")
        return

    print(f"Loaded {len(documents)} document pages/sections.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")

    # Create embeddings
    print("Initializing local Hugging Face embeddings (this may download the model)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} 
    )

    print(f"Creating vector store in {persist_directory}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Vector store created successfully.")
    return vector_store


def create_rag_chain(persist_directory):
    """Create a simple retrieval-augmented generation chain."""
    print("Initializing RAG chain...")

    groq_api_key = os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b", 
        temperature=0.2,
        groq_api_key=groq_api_key  # <-- Add this line
    )
    
    print("Initializing local Hugging Face embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    print(f"Loading vector store from {persist_directory}...")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print("Retriever ready.")

    prompt_template = """
    You are a helpful assistant for answering questions using context.
    Use the following retrieved context to answer the user's question.
    If you don't know the answer, just say you don't know.
    
    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        RunnableMap({
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain created successfully.")
    return rag_chain


if __name__ == "__main__":
    import sys

    persist_dir = "vector_store"

    if len(sys.argv) > 2:
        command = sys.argv[1]

        if command == "index":
            file_to_index = sys.argv[2]
            if not os.path.exists(file_to_index):
                print(f"Error: File not found at {file_to_index}")
            else:
                create_vector_store(file_to_index, persist_dir)

        elif command == "query":
            query_text = sys.argv[2]
            print(f"Querying: '{query_text}'")
            chain = create_rag_chain(persist_dir)
            answer = chain.invoke(query_text)
            print("\n--- Answer ---")
            print(answer)

    else:
        print("Usage:")
        print("  To index: python rag_pipeline.py index \"<path_to_file.pdf>\"")
        print("  To query: python rag_pipeline.py query \"<your_question>\"")