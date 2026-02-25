import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
import os
import base64

st.set_page_config(page_title="Chat with Ragavi", page_icon="💼", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

/* Force white background on main area */
.stApp { background-color: #f5f7fa !important; }

/* Sidebar dark */
[data-testid="stSidebar"] {
    background: #1e293b !important;
    border-right: 2px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Profile */
.profile-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 30px 16px 16px;
}
.profile-img-container {
    width: 110px;
    height: 110px;
    border-radius: 50%;
    overflow: hidden;
    border: 3px solid #3b82f6;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.2);
    margin-bottom: 14px;
    flex-shrink: 0;
}
.profile-img-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: top center;
}
.profile-name {
    color: #f1f5f9 !important;
    font-size: 20px;
    font-weight: 700;
    margin: 0 0 4px;
    text-align: center;
}
.profile-role {
    color: #93c5fd !important;
    font-size: 13px;
    text-align: center;
    margin: 0;
}

/* Suggestion buttons */
.stButton > button {
    background: #334155 !important;
    color: #e2e8f0 !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    width: 100% !important;
    text-align: left !important;
    margin-bottom: 6px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #3b82f6 !important;
    border-color: #3b82f6 !important;
    color: #fff !important;
}

/* Main chat area */
.main-header {
    background: white;
    padding: 24px 28px 20px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.main-header h1 {
    color: #0f172a !important;
    font-size: 26px;
    font-weight: 700;
    margin: 0 0 4px;
}
.main-header p {
    color: #64748b !important;
    font-size: 14px;
    margin: 0;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
[data-testid="stChatMessage"] p {
    color: #1e293b !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: white !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #3b82f6 !important;
}
</style>
""", unsafe_allow_html=True)

# ── API + DB ───────────────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_db():
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)

db = load_db()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    try:
        with open("profile.jpg", "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        img_html = f'<img src="data:image/jpeg;base64,{img_b64}" />'
    except:
        img_html = '<div style="width:100%;height:100%;background:#334155;display:flex;align-items:center;justify-content:center;font-size:40px;">👤</div>'

    st.markdown(f"""
    <div class="profile-wrapper">
        <div class="profile-img-container">{img_html}</div>
        <p class="profile-name">Ragavi</p>
        <p class="profile-role">UX Designer</p>
    </div>
    <hr style="border-color:#334155;margin:8px 0 16px;">
    <p style="color:#94a3b8 !important;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;padding:0 4px;margin-bottom:10px;">💡 Suggested Questions</p>
    """, unsafe_allow_html=True)

    suggestions = [
        "👩‍💻 What are your top skills?",
        "💼 Tell me about your experience",
        "🚀 What projects have you built?",
        "🎓 Educational background?",
        "🌟 What makes you stand out?",
    ]
    for s in suggestions:
        if st.button(s, key=s):
            st.session_state.pending_question = s

# ── Main ───────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>💼 Chat with Ragavi's Resume</h1>
    <p>Ask me anything about my background, skills, and experience</p>
</div>
""", unsafe_allow_html=True)

SYSTEM_CONTEXT = """
You are a professional assistant representing Ragavi.
Answer questions about their skills and experience in a friendly,
professional tone using information from the resume context provided.
If something isn't in the resume, say so honestly.
Keep answers concise and well-structured.
"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! I'm Ragavi's resume assistant. Ask me anything about her skills, experience, or background!"}
    ]
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Ask about skills, experience, projects...")
if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking..."):
        docs = db.similarity_search(prompt, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        response = llm.invoke([
            SystemMessage(content=SYSTEM_CONTEXT),
            HumanMessage(content=f"Resume context:\n{context}\n\nQuestion: {prompt}")
        ])
        answer = response.content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
    st.rerun()