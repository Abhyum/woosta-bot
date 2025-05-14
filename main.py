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
import google.generativeai as genai


# === Configuration ===
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment variables.")
os.environ["GOOGLE_API_KEY"] = "AIzaSyDzOkE_VZMG7kzSuuGsoPpTP7_bExVKOfE"
genai.configure(api_key=api_key)
model1 = genai.GenerativeModel("gemini-2.0-flash")
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
    print("test")
    docs = retriever.get_relevant_documents(q.question)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    result = chain({"input_documents": docs, "question": q.question}, return_only_outputs=True)
    print( "sadasds\n", q.question , result['output_text'])
    response = model1.generate_content(f"""
    You are Woostaa, a smart, professional assistant for a home services platform that offers on-demand services like cleaning, repair, and maintenance, typically charged by the hour.

    Here is the customer's question: "{q.question}"
    Here is the current answer: "{result["output_text"]}"

    Instructions:
    1. If the question is a greeting (e.g., "hi", "hello"), reply warmly and briefly like a human.
    2. If the answer says something like "answer not available in context", politely say you're here to help and can assist with anything related to home services.
    3. Otherwise, rewrite the answer to be:
    - Short and natural
    - Friendly, but professional
    - Clear and helpful, like a support rep
    - well dont give respinse in md just in plain text okey
    - aslo properly ans  hello , howare you thing

    Respond with just the final response text.
    """)

    # print(response.text)

    return {"answer": response.text }


# === Serve chatbot UI ===
@app.get("/")
def serve_chatbot():
    return FileResponse("Static/chatbotwosta.html")
