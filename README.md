# PDF Chat — RAG Pipeline with FastAPI

A production-style Retrieval-Augmented Generation (RAG) system that allows users to upload any PDF and ask questions against it. Built with LangChain, FastAPI, and Groq.

---

## Architecture

```
PDF Upload
    └── PyMuPDFLoader         → extracts text from PDF
    └── RecursiveCharacterTextSplitter → chunks the text
    └── HuggingFaceEmbeddings → converts chunks to vectors
    └── Chroma (in-memory)    → stores and indexes vectors

Query
    └── MMR Retriever         → fetches top-k relevant chunks
    └── ContextualCompressionRetriever → removes noise from retrieved chunks
    └── RunnableParallel      → passes query and compressed context simultaneously
    └── PromptTemplate        → structures the input for the LLM
    └── llama-3.3-70b         → generates the final answer
    └── StrOutputParser       → returns clean string output
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| API Framework | FastAPI |
| Document Loader | PyMuPDF |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | Chroma |
| Retrieval Strategy | MMR (Maximal Marginal Relevance) |
| Compression | LLMChainExtractor |
| LLM (compression) | llama-3.1-8b-instant via Groq |
| LLM (answer) | llama-3.3-70b-versatile via Groq |
| Chain Orchestration | LangChain LCEL |

---

## Why These Choices

**MMR over similarity search** — standard similarity search returns the most similar chunks which are often redundant. MMR balances relevance with diversity, ensuring the retrieved context covers more ground.

**Contextual Compression** — raw retrieved chunks contain a lot of text irrelevant to the query. The compressor uses a smaller LLM to extract only the relevant portion before passing it to the main model. This reduces noise and keeps the context window clean.

**Two LLMs** — a smaller model (8b) handles compression where speed matters, and a larger model (70b) handles the final answer where quality matters.

**Chroma in-memory with `delete_collection()`** — each new PDF upload clears the previous vectorstore entirely, ensuring no cross-document contamination.

---

## API Endpoints

### `POST /upload`
Upload a PDF file for processing.

**Request:** `multipart/form-data` with a `file` field.

**Response:**
```json
{
  "filename": "document.pdf",
  "total_pages": 19,
  "total_chunks": 87,
  "status": "ready to answer"
}
```

---

### `POST /ask`
Ask a question against the uploaded PDF.

**Query param:** `?query=your question here`

**Response:**
```json
{
  "question": "What is the main topic?",
  "answer": "..."
}
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/pdf-chat.git
cd pdf-chat
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**
```bash
cp .env.example .env
```
Open `.env` and add your Groq API key. Get one free at [console.groq.com](https://console.groq.com).

**5. Run the server**
```bash
uvicorn summarizer:app --reload
```

Server runs at `http://localhost:8000`.
Interactive API docs available at `http://localhost:8000/docs`.

---

## Frontend

A minimal HTML/CSS frontend is included (`index.html`) with a 70/30 split layout — PDF viewer on the left, chat interface on the right.

To serve it locally:
```bash
cd ui
python -m http.server 3000
```

Open `http://localhost:3000/index.html` in your browser.

---

## Project Structure

```
pdf-chat/
├── summarizer.py       # FastAPI app and RAG pipeline
├── ui/
│   └── index.html      # Frontend
├── file/               # Uploaded PDFs (gitignored)
├── .env                # API keys (gitignored)
├── .env.example        # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Notes

- The vectorstore is in-memory. Restarting the server clears it and requires re-uploading the PDF.
- The embedding model downloads on first run and is cached locally by HuggingFace.
- GPU acceleration is supported — set `model_kwargs={"device": "cuda"}` in `HuggingFaceEmbeddings` if available.
