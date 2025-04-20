# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with actual frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + PDF once
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

pdf_path = "woostaachat.pdf"  # Must exist in repo root
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
texts = text_splitter.split_text("\n\n".join([p.page_content for p in pages]))

retriever = Chroma.from_texts(texts, embeddings).as_retriever()

prompt_template = """Answer the question as precise as possible using the provided context. 
If the answer is not contained in the context, say "answer not available in context".

Context: 
{context}

Question: 
{question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    docs = retriever.get_relevant_documents(q.question)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    result = chain({"input_documents": docs, "question": q.question}, return_only_outputs=True)
    return {"answer": result["output_text"]}
