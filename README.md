# RAG-MultiFile-QA
![GitHub Repo stars](https://img.shields.io/github/stars/Uni-Creator/RAG-MultiFile-QA?style=social)  ![GitHub forks](https://img.shields.io/github/forks/Uni-Creator/RAG-MultiFile-QA?style=social)

📚 **Multi-File Retrieval-Augmented Generation (RAG) Q&A System**

This project is a **Streamlit-based Q&A application** that allows users to upload multiple document types (**PDF, DOCX, TXT, CSV**) and ask questions about their content using **retrieval-augmented generation (RAG)**.

## 🔹 Features
- Upload and process multiple files at once.
- Supports **PDF, DOCX, TXT, and CSV** formats.
- Uses **Hugging Face Embeddings** and **FAISS vector search** for document retrieval.
- Integrates **Hugging Face Inference API** for generating responses.
- Maintains **chat history** for seamless user experience.
- **Clear all** button to reset uploaded files and chat history.

## 🛠️ Tech Stack
- **Python**
- **Streamlit** (Frontend UI)
- **Langchain** (Document Processing & Retrieval)
- **Hugging Face Inference API** (LLM-based Answer Generation)
- **FAISS** (Vector Store for Efficient Retrieval)
- **PyPDFLoader, TextLoader, CSVLoader** (File Parsing)

## 🚀 How to Run
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/RAG-MultiFile-QA.git
   cd RAG-MultiFile-QA
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set your **Hugging Face API Key** as an environment variable:
   ```sh
   export HUGGINGFACE_API_KEY="your_api_key"
   ```
4. Run the app:
   ```sh
   streamlit run main.py
   ```

## 📌 Notes
- Ensure your **Hugging Face API Key** is correctly set.
- The system works best with **structured documents** containing well-defined sections and tables.
- **FAISS indexing** helps in faster search and retrieval from large documents.

## Execution Requirements

-This project is designed to run locally as a Streamlit web application. It requires Ollama to be installed and running on the local machine to perform Large Language Model inference. The default model used in this project is gemma3:4b, and the Ollama server is expected to be accessible at http://localhost:11434 during execution.

## 📜 License
This project is **open-source** and available under the **MIT License**.

## 🔗 Project Origin & Credits

This project is **based on and inspired by** the following repository:

- Original repository: https://github.com/Uni-Creator/RAG-MultiFile-QA

### Modifications & Extensions

The original project has been **significantly extended and refactored** with the following features:

- **Local LLM integration using Ollama** (no paid APIs required)
- **Multilingual support** (TR / EN / FR / ES)
- **Voice input** using offline Whisper (faster-whisper)
- **Voice output** using offline TTS
- **Device-aware execution** (CPU / CUDA / Apple Silicon MPS)
- **Device-aware embeddings** (GPU / MPS acceleration when available)
- **Adaptive Whisper runtime selection**  
  - CUDA → GPU + float16  
  - Apple Silicon → CPU + int8 (stability-focused)
- Improved Streamlit UI and interaction flow
- Refactored RAG pipeline and prompt handling
- How to Run the Project:
  1-) Firstly "cd RAG-MultiFile-QandA" command in terminal for moving into the correct project directory
  2-) Secondly "streamlit run main.py" command in terminal for launching the Streamlit web application


  All modifications and enhancements were implemented by the authors of this repository.
