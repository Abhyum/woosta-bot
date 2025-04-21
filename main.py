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

# === Configuration ===
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment variables.")
os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

app = FastAPI()

# === Serve static files ===
app.mount("/Static", StaticFiles(directory="Static"), name="Static")

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["https://woostaa.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load the model and embeddings ===
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# === Load and process PDF ===
pdf_path = "woostaachat.pdf"
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

# === Pydantic model for request ===
class Question(BaseModel):
    question: str

# === POST endpoint for answering questions ===
@app.post("/ask")
def ask(q: Question):
    docs = retriever.get_relevant_documents(q.question)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    result = chain({"input_documents": docs, "question": q.question}, return_only_outputs=True)
    return {"answer": result["output_text"]}

# === Serve chatbot UI ===
@app.get("/")
def serve_chatbot():
    return FileResponse("Static/chatbotwosta.html")
