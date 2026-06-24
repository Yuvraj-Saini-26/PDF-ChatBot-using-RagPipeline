from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import shutil
import os

load_dotenv()

os.makedirs("file", exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mainchain = None
vectorstore = None
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm_small = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
llm_main = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))


def formator(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


prompt = PromptTemplate(
    input_variables=['context', 'query'],
    template="""
You are a precise document assistant. Answer based ONLY on the context below.

Context:
{context}

Question: {query}

Instructions:
- Give a well structured answer with clear sections if needed
- Use bullet points for lists or multiple items
- Use numbered steps for processes or sequences
- Bold important terms using **term** syntax
- Keep answers concise but complete
- If answer is NOT in context, say exactly: "I don't know."
- Do NOT add outside information

Answer:"""
)


@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    global mainchain, vectorstore

    save_path = f"file/{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyMuPDFLoader(save_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    if vectorstore is not None:
        vectorstore.delete_collection()

    vectorstore = Chroma.from_documents(documents=split_docs, embedding=model)

    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 4, 'fetch_k': 10, 'lambda_mult': 0.3}
    )

    compressor = LLMChainExtractor.from_llm(llm_small)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    chain01 = RunnableParallel({
        'context': compression_retriever | RunnableLambda(formator),
        'query': RunnablePassthrough()
    })

    mainchain = chain01 | prompt | llm_main | StrOutputParser()

    return {
        "filename": file.filename,
        "total_pages": len(docs),
        "total_chunks": len(split_docs),
        "status": "ready to answer"
    }


@app.post('/ask')
def ask(query: str):
    if mainchain is None:
        return JSONResponse(status_code=400, content={"error": "Upload a PDF first."})
    answer = mainchain.invoke(query)
    return JSONResponse(status_code=200, content={"question": query, "answer": answer})
