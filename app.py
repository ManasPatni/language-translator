import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st

# 🛠️ Set page config FIRST, before any other Streamlit function
st.set_page_config(page_title="🌍 AI Translator Chatbot", layout="centered")

# Now, proceed with other imports and Streamlit commands
st.title("Welcome to AI Translator Chatbot")


import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util

# ✅ Initialize ChromaDB for Translation Memory
@st.cache_resource
def initialize_chromadb():
    """Initialize ChromaDB client and create/retrieve a collection for translations."""
    chroma_client = chromadb.PersistentClient(path="./chroma_translation_db")
    collection = chroma_client.get_or_create_collection(name="translation_memory")
    return collection

collection = initialize_chromadb()

# ✅ Load HuggingFace Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize Sentence-Transformers Model for Semantic Matching
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Initialize Llama 3 for Language Translation
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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")

# ✅ Query Llama 3 for Translation
def query_llama3(text, source_lang, target_lang):
    system_prompt = f"""
    System Prompt: You are 'Polyglot Translator', an AI that translates text fluently between multiple languages.

    Instructions:
    - Translate text while keeping the original meaning intact.
    - Ensure grammar accuracy and natural fluency.
    - Maintain translation consistency using past translations.
    - Provide clear and natural translations.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"🔤 Translate '{text}' from {source_lang} to {target_lang}:")
    ]

    try:
        response = chat.invoke(messages)
        return response.content
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# ✅ Streamlit UI
st.set_page_config(page_title="🌍 AI Translator Chatbot", layout="centered")

st.title("🌍 AI Translator Chatbot")
st.write("Translate text fluently between multiple languages using AI-powered translation memory!")

# User Input
user_text = st.text_area("🔤 Enter text to translate:", "")
source_language = st.text_input("💬 Source Language (e.g., 'en' for English, 'fr' for French):", "")
target_language = st.text_input("💬 Target Language:", "")

if st.button("Translate"):
    if user_text and source_language and target_language:
        with st.spinner("🔄 Translating..."):
            translated_text = query_llama3(user_text, source_language, target_language)
        
        st.success("✅ Translation Completed!")
        st.subheader(f"📝 Translation ({target_language}):")
        st.write(translated_text)
    else:
        st.warning("⚠️ Please enter text, source language, and target language.")

# Footer
#st.markdown("---")
#st.markdown("Made with ❤️ using **Llama 3, ChromaDB & Streamlit**")
