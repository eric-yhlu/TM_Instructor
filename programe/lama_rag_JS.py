from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from llama_index.core import Document, VectorStoreIndex
from langchain.text_splitter import CharacterTextSplitter  
from langchain_openai import ChatOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import re
import os
from dotenv import load_dotenv 

app = Flask(__name__, static_folder="../GUI")
CORS(app)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model='gpt-4o')

# Dictionary to store memory for each conversation
conversation_memory = {}

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text

def load_pdf_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    documents = []
    for doc in docs:
        cleaned_content = clean_text(doc.page_content)
        chunks = text_splitter.split_text(cleaned_content)
        for chunk in chunks:
            documents.append(Document(text=chunk))
    return documents

def summarize_memory(memory):
    response = llm.invoke(f"摘要：{memory}")
    return response.content

def generate_clarifying_question(input_text):
    response = llm.invoke(f"問題：{input_text} 過於模糊，請根據問題生成更聚焦的澄清性問題，幫助使用者提供更多具體信息。")
    return response.content

pdf_file_path = "programe/document/GSAF21091303_Service_Manual_of_HW3.2_for_TMAA_and_TMAAM_TW.pdf"  
pdf_documents = load_pdf_document(pdf_file_path)

embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")
index = VectorStoreIndex.from_documents(pdf_documents, embeddings=embedding_model)
retriever = index.as_retriever()

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'login.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/get-response', methods=['POST'])
def ask_question():
    data = request.json
    input_text = data.get("question", "")
    conversation_id = data.get("conversationId", "default")
    print('-'*50)
    print("conversation_id: ", conversation_id)
    # Ensure the conversation ID has an associated memory
    if conversation_id not in conversation_memory:
        conversation_memory[conversation_id] = []

    memory = conversation_memory[conversation_id]

    retrieved_docs = retriever.retrieve(input_text)
    print('-'*50)
    
    if retrieved_docs:
        context = "\n".join([doc.text for doc in retrieved_docs])  
        response = llm.invoke(f"""以下內容為搜索到的東西：\n\n{context}\n\n使用者問題：{input_text}，請根據使用者的問題去搜索到的內容找答案，
                              若根據搜索到的內容可以找出答案，請直接回答問題
                              若你認為使用者問題不明確，請輸出一點提示，幫助使用者收斂問題；
                              **使用者問題不明確的定義舉例:假如內容中提到電控箱檢修有分成電源開啟時以及關閉時，若使用者只提問電控箱的檢修流程，
                              即為問題不清楚，要想辦法引導使用者說出他要哪種電控箱檢查流程。若有需要，你可以參考之前對話記憶：\n\n{memory}
                              """)
        #response = llm.invoke(f"根據以下內容回答問題，若有需要可以參考先前對話記憶：\n\n{context}\n\n問題：{input_text}；\n\n記憶：{memory}")
        
        memory.append(f"問題：{input_text}\n回答：{response.content}\n")
        
        # Limit memory size for each conversation to manage memory usage
        memory_len = sum(len(m) for m in memory)
        if memory_len > 1000:
            recent_memory = memory[-5:] 
            summary = summarize_memory(" ".join(memory[:-5]))
            if summary:
                memory = [summary] + recent_memory
                conversation_memory[conversation_id] = memory
        print("memory: ", memory)
        print(response.content)
    else:
        clarifying_question = generate_clarifying_question(input_text)
        print(f"您的問題過於模糊，請提供更多具體信息。您可以嘗試：{clarifying_question}")

    if retrieved_docs:
        formatted_response = response.content.replace('\n', '\\n')
        return jsonify({'answer': formatted_response})
    
    
    else:
        formatted_clarifying_question = clarifying_question.replace('\n', '\\n')
        return jsonify({"answer": f"您的問題過於模糊，請提供更多具體信息。您可以嘗試：{formatted_clarifying_question}"})
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
