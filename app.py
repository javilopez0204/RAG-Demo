import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Configuración de la página
st.set_page_config(page_title="SafeBank Soporte", page_icon="🏦")
st.title("🏦 SafeBank - Asistente Virtual")

# 2. Cargar API Key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.info("Por favor, configura tu GROQ_API_KEY en el archivo .env")
    st.stop()

# 3. Función de Cache para RAG (Carga el PDF solo una vez)
@st.cache_resource
def initialize_rag():
    file_path = "safebank-manual.pdf"
    if not os.path.exists(file_path):
        return None
    
    # Cargar documento
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # Dividir texto (mismos parámetros que tu notebook)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    # Crear Embeddings y Vector Store
    # Usamos el modelo que definiste en el notebook
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = initialize_rag()

if retriever is None:
    st.error("No se encontró el archivo 'safebank-manual.pdf'. Por favor asegúrate de que esté en la carpeta.")
    st.stop()

# 4. Configurar el LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_retries=2,
    api_key=api_key
)

# 5. Configurar el Prompt (Basado en tu notebook)
system_prompt = """You are a helpful virtual assistant answering general questions about a company's services called SafeBank.
Use the following bits of retrieved context to answer the question.
If you don't know the answer, just say you don't know. Keep your answer concise.
Context: {context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Crear la cadena RAG
rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# 6. Interfaz de Chat (Con historial)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar input del usuario
if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Consultando el manual..."):
            try:
                response = rag_chain.invoke(prompt)
                
                # Limpiar tags de pensamiento si aparecen (DeepSeek/Llama thinking tags)
                if "</think>" in response:
                    response = response.split("</think>")[-1].strip()
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")