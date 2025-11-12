import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()
from rag_pipeline import create_vector_store, create_rag_chain

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
PERSIST_DIRECTORY = 'vector_store'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and ingestion into the vector store."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
            print(f"Starting vector store creation for: {file_path}")
            create_vector_store(file_path, PERSIST_DIRECTORY)
            print("Vector store creation complete.")            
            return jsonify({"status": "success", "filename": filename}), 201
            
        except Exception as e:
            print(f"Error during file upload or processing: {e}")
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/query', methods=['POST'])
def query_document():
    """Handles user queries against the ingested document."""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
        
    query = data['query']
    
    try:
        print("Creating RAG chain...")
        rag_chain = create_rag_chain(PERSIST_DIRECTORY)
    
        print(f"Invoking chain with query: {query}")
        answer = rag_chain.invoke(query)
        return jsonify({"answer": answer}), 200
        
    except Exception as e:
        print(f"Error during query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    """Serves a default favicon to prevent 404 errors in the log."""
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)