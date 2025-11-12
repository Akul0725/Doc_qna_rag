# 🧠 Document Q&A System with RAG Pipeline

A **web application** that enables users to **upload documents (PDF or text)** and instantly **query their content** for summaries or specific answers using an advanced **Retrieval-Augmented Generation (RAG)** pipeline.

---

## 🚀 Overview

This project implements a complete **RAG (Retrieval-Augmented Generation)** pipeline integrated into a Flask web app.

Users can:
- Upload documents (`.pdf`, `.txt`, etc.)
- Ask questions or request summaries
- Get intelligent answers generated from their own document’s content

---

## 🧩 Features

✅ Upload and process multiple documents  
✅ Summarize or query document contents instantly  
✅ Modular RAG pipeline using **LangChain**  
✅ Embeddings-based vector search (FAISS or ChromaDB)  
✅ Real-time API endpoints via **Flask**  
✅ Supports any LLM backend integration  

---

## 🏗️ System Architecture

**RAG Pipeline Flow:**

1. **Document Upload** → User uploads PDF/Text file.  
2. **Document Loader** → Extracts and cleans content.  
3. **Text Splitter** → Chunks text into manageable pieces.  
4. **Embedding Model** → Converts chunks into vector embeddings.  
5. **Vector Store (FAISS/Chroma)** → Stores and indexes embeddings.  
6. **Retriever + LLM** → Retrieves relevant chunks and generates answers.  
7. **Response API** → Returns the summarized or queried answer to the user.

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| Backend | **Python**, **Flask** |
| ML/NLP Framework | **LangChain** |
| Vector Store | **FAISS** / **ChromaDB** |
| Embeddings | **Sentence Transformers** / **OpenAI Embeddings** |
| Frontend (Optional) | HTML + Templates |
| Other Tools | `pdfplumber`, `PyMuPDF`, `dotenv`, `requests` |

---

## 📂 Project Structure

Document-QA-RAG
│
├── templates/ # HTML templates for web UI
├── app.py # Flask app entry point
├── rag_pipeline.py # Core RAG logic (loaders, retrievers, embeddings)
├── requirements.txt # All dependencies
├── .gitignore # Ignored files

Create a Virtual Environment
python -m venv venv
source venv/bin/activate    # (Linux/Mac)
venv\Scripts\activate       # (Windows)

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Flask App
python app.py


The app will start on http://127.0.0.1:5000/ by default.


🧠 Usage


Open the app in your browser.


Upload a PDF or text file.


Enter your question in the input box (e.g., “Summarize this document” or “What are the main topics discussed?”).


View instant AI-generated answers retrieved from your own document.



🧪 Example Queries


“Summarize the uploaded document.”


“What is the key finding of section 3?”


“List all entities mentioned in the report.”


“Who are the main stakeholders discussed?”


🧑‍💻 Author
Akul Kalia
💼 GitHub: Akul0725

📜 License
This project is licensed under the MIT License — feel free to use and modify.


