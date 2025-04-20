from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

# === CONFIGURATION ===
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

app = FastAPI()

# === Serve Static Files (HTML, JS, CSS) ===
app.mount("/static", StaticFiles(directory="static"), name="static")

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Model and PDF Once ===
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

pdf_path = "woostaachat.pdf"  # PDF must be in root directory
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
texts = text_splitter.split_text("\n\n".join([p.page_content for p in pages]))

retriever = Chroma.from_texts(texts, embeddings).as_retriever()

# === Prompt Template ===
prompt_template = """Answer the question as precise as possible using the provided context. 
If the answer is not contained in the context, say "answer not available in context".

Context: 
{context}

Question: 
{question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# === Request Model ===
class Question(BaseModel):
    question: str

# === Chat API Endpoint ===
@app.post("/ask")
def ask(q: Question):
    docs = retriever.get_relevant_documents(q.question)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    result = chain({"input_documents": docs, "question": q.question}, return_only_outputs=True)
    return {"answer": result["output_text"]}

# === Route to Serve Chatbot HTML ===
@app.get("/")
def serve_chatbot():
    return FileResponse("static/chatbotwosta.html")
