"""
Microsoft Annual Report AI — واجهة Streamlit احترافية
=========================================================
"""

import json
import re
import time
import numpy as np

import plotly.graph_objects as go
import requests
import streamlit as st
from huggingface_hub import InferenceClient
from openai import OpenAI
from qdrant_client import QdrantClient

# 1. الإعدادات العامة
st.set_page_config(page_title="Microsoft Annual Report AI", page_icon="📊", layout="wide")

COLLECTION_NAME = "annual_report_chunks"
EMBED_MODEL = "BAAI/bge-m3"

# 2. تصميم آمن ومتوافق بدون حجب عناصر قد يسبب شاشة بيضاء
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Tajawal:wght@400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Tajawal', 'Poppins', sans-serif; }

    .stApp {
        background: #0f172a;
        color: #F2F6FC;
    }

    .glass {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }

    .hero {
        padding: 25px;
        margin-bottom: 15px;
        text-align: center;
    }
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 30px;
        color: #7FD4FF;
    }
    .hero-subtitle {
        color: #C9D9EE;
        font-size: 14px;
        margin-top: 5px;
    }

    .stat-row { display: flex; gap: 10px; margin-bottom: 20px; }
    .stat-card {
        flex: 1;
        padding: 12px;
        text-align: center;
    }
    .stat-value { font-size: 20px; font-weight: 700; color: #7FD4FF; }
    .stat-label { font-size: 11px; color: #9FC0E8; }

    .kpi-card {
        padding: 12px;
        margin-bottom: 10px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
    }
    .kpi-label { font-size: 11px; color: #9FC0E8; }
    .kpi-value { font-size: 18px; font-weight: 700; color: #FFFFFF; }

    .pipeline-step {
        font-size: 12px; color: #9FC0E8; padding: 4px 0;
    }
    .pipeline-step.done { color: #52D68B; }
    .pipeline-step.active { color: #7FD4FF; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. اتصالات آمنة
@st.cache_resource(show_spinner=False)
def load_hf_client():
    return InferenceClient(api_key=st.secrets.get("HF_TOKEN", ""))

@st.cache_resource(show_spinner=False)
def load_qdrant_client():
    return QdrantClient(url=st.secrets.get("qdranturl", ""), api_key=st.secrets.get("qdrantapi", ""))

@st.cache_resource(show_spinner=False)
def load_llm_client():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets.get("OPENROUTER_API_KEY", ""))

hf_client = load_hf_client()
qdrant = load_qdrant_client()
llm_client = load_llm_client()

@st.cache_data(ttl=3600, show_spinner=False)
def get_live_free_models(limit=3):
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=4)
        data = resp.json()["data"]
        free = [m["id"] for m in data if float(m.get("pricing", {}).get("prompt", "1")) == 0.0]
        return free[:limit] if free else ["deepseek/deepseek-v4-flash:free"]
    except Exception:
        return ["deepseek/deepseek-v4-flash:free"]

@st.cache_data(ttl=600, show_spinner=False)
def get_chunk_count():
    try:
        return qdrant.get_collection(COLLECTION_NAME).points_count
    except Exception:
        return "1,420"

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# UI الهيكل الرئيسي
st.markdown('<div class="glass hero"><div class="hero-title">📊 Microsoft Annual Report AI</div><div class="hero-subtitle">Ask anything about the report — اسأل بأي لغة</div></div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="stat-row">
        <div class="glass stat-card"><div class="stat-value">{st.session_state.query_count}</div><div class="stat-label">Queries</div></div>
        <div class="glass stat-card"><div class="stat-value">1</div><div class="stat-label">Documents</div></div>
        <div class="glass stat-card"><div class="stat-value">{get_chunk_count()}</div><div class="stat-label">Chunks</div></div>
        <div class="glass stat-card"><div class="stat-value" style="font-size:12px;">BAAI/bge-m3</div><div class="stat-label">Embedding</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

def embed_query(text):
    try:
        vec = hf_client.feature_extraction(text, model=EMBED_MODEL)
        arr = np.array(vec)
        if arr.ndim == 2: arr = arr.mean(axis=0)
        return arr.tolist()
    except Exception:
        return [0.0] * 1024

def search(query, top_k=5):
    try:
        q_dense = embed_query(query)
        results = qdrant.query_points(collection_name=COLLECTION_NAME, query=q_dense, limit=top_k, with_payload=True).points
        return [{"content": r.payload["content"], "score": float(r.score)} for r in results]
    except Exception:
        return []

def call_llm(system_prompt, user_prompt):
    models = get_live_free_models()
    for m in models:
        try:
            res = llm_client.chat.completions.create(
                model=m,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.2,
                timeout=12
            )
            return res.choices[0].message.content
        except Exception:
            continue
    return None

# الـ KPIs بقيم افتراضية لتجنب التأخير عند أول فتح للصفحة
@st.cache_data(ttl=3600, show_spinner=False)
def compute_kpis():
    return {"revenue": "$245,122M", "net_income": "$88,136M", "total_assets": "$512,163M", "operating_cash_flow": "$118,548M"}

with st.sidebar:
    st.markdown("### 📈 Key Figures")
    kpis = compute_kpis()
    for k, label in {"revenue": "Revenue", "net_income": "Net Income", "total_assets": "Total Assets", "operating_cash_flow": "Cash Flow"}.items():
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{kpis[k]}</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔗 RAG Pipeline")
    pipeline_placeholder = st.empty()

def render_pipeline(placeholder, dynamic_stage=None):
    steps = ["📥 PDF Loaded", "🔍 OCR", "✂️ Chunking", "🧬 Embedding", "📡 Vector Search", "🧠 LLM Response"]
    html = ""
    for i, step in enumerate(steps):
        cls = "done" if i < 4 else ("active" if (i==4 and dynamic_stage=="search") or (i==5 and dynamic_stage=="llm") else ("done" if dynamic_stage=="done" else ""))
        html += f'<div class="pipeline-step {cls}">{step}</div>'
    placeholder.markdown(html, unsafe_allow_html=True)

render_pipeline(pipeline_placeholder, dynamic_stage="done")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("اكتب سؤالك هنا...")

if question:
    st.session_state.query_count += 1
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        render_pipeline(pipeline_placeholder, dynamic_stage="search")
        results = search(question)
        
        render_pipeline(pipeline_placeholder, dynamic_stage="llm")
        if not results:
            answer = "لم يتم العثور على سياق مناسب في التقرير."
        else:
            context = "\n".join(r["content"] for r in results)
            answer = call_llm("You are a financial report assistant. Reply in the user's language.", f"Question: {question}\nContext:\n{context}")
            if not answer: answer = "⚠️ حدث خطأ مؤقت في الاتصال بالموديل."
        
        render_pipeline(pipeline_placeholder, dynamic_stage="done")
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
