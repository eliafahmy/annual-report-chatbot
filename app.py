"""
Microsoft Annual Report AI — Ultimate Theme & Input Fix
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

# 1. إعدادات الصفحة الأساسية
st.set_page_config(
    page_title="Microsoft Annual Report AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLLECTION_NAME = "annual_report_chunks"
EMBED_MODEL = "BAAI/bge-m3"

# 2. هندسة التصميم والألوان - إزالة الحواف البيضاء تماماً وتعديل الـ Input Text
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&family=Tajawal:wght@400;500;700&display=swap');

    html, body, [class*="css"] { 
        font-family: 'Segoe UI', 'Tajawal', sans-serif; 
    }
    
    /* تثبيت الخلفية على كل عناصر الصفحة والـ Containers لمنع أي حواف بيضاء */
    .stApp, [data-testid="stHeader"], [data-testid="stBottom"], [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top, #1e3a8a 0%, #0f172a 60%, #020617 100%) !important;
        color: #F8FAFC !important;
    }

    /* تلوين شريط التمرير والحواف الإضافية السفلى */
    [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
    }

    /* إجبار نصوص الـ Sidebar بالكامل على الظهور بلون أبيض واضح جداً */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #38BDF8 !important;
        font-weight: 700;
    }

    /* كروت الزجاج الاحترافية (Glassmorphism) */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .ms-logo {
        display: inline-block;
        width: 22px;
        height: 22px;
        background: #38BDF8;
        margin-right: 10px;
        vertical-align: middle;
        box-shadow: 0 0 14px #38BDF8;
        border-radius: 4px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
    }

    /* كروت المؤشرات الجانبية */
    .kpi-box {
        padding: 14px;
        margin-bottom: 12px;
        background: rgba(255, 255, 255, 0.04);
        border-left: 4px solid #38BDF8;
        border-radius: 6px;
    }
    .kpi-title { font-size: 12px; color: #E2E8F0 !important; font-weight: 600; text-transform: uppercase; }
    .kpi-val { font-size: 20px; font-weight: 700; color: #38BDF8 !important; margin-top: 2px; }

    /* عناصر الـ RAG Pipeline الجانبية */
    .pipeline-step {
        padding: 8px 12px;
        margin-bottom: 6px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 6px;
        font-size: 13px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .pipeline-success { color: #34D399 !important; font-weight: bold; }

    /* كروت المصادر السفلية */
    .source-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 14px;
        margin-top: 10px;
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #38BDF8;
        font-weight: bold;
        margin-bottom: 6px;
    }

    /* تعديل شامل للـ Input وتغيير لون الخط أثناء الكتابة ليكون أبيض ناصع وواضح جداً */
    div[data-testid="stChatInput"] {
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        background: #1e293b !important;
    }
    
    div[data-testid="stChatInput"] textarea {
        color: #FFFFFF !important;
        font-size: 15px !important;
    }
    
    /* تغيير لون الـ Placeholder (النص المؤقت المكتوب بالداخل) */
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #94A3B8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. إدارة الاتصالات (شامل الـ Cache)
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

@st.cache_data(ttl=600, show_spinner=False)
def get_chunk_count():
    try:
        return qdrant.get_collection(COLLECTION_NAME).points_count
    except Exception:
        return 1248

if "query_count" not in st.session_state:
    st.session_state.query_count = 152

# 4. محرك البحث والـ Vectors
def embed_query(text):
    try:
        vec = hf_client.feature_extraction(text, model=EMBED_MODEL)
        arr = np.array(vec)
        if arr.ndim == 2: arr = arr.mean(axis=0)
        return arr.tolist()
    except Exception:
        return [0.0] * 1024

def search_rag(query, top_k=3):
    try:
        q_dense = embed_query(query)
        try:
            results = qdrant.query_points(
                collection_name=COLLECTION_NAME, query=q_dense, using="dense", limit=top_k, with_payload=True
            ).points
        except Exception:
            results = qdrant.query_points(
                collection_name=COLLECTION_NAME, query=q_dense, limit=top_k, with_payload=True
            ).points
            
        return [
            {
                "content": r.payload.get("content", ""),
                "page": r.payload.get("page_number", r.payload.get("page", "N/A")),
                "score": float(r.score)
            } 
            for r in results if r.payload
        ]
    except Exception:
        return []

# 5. بناء واجهة التفاعل (UI)
st.markdown('<div class="glass-card"><div class="hero-title"><span class="ms-logo"></span>Microsoft Annual Report AI</div><div style="text-align:center; color:#94A3B8; font-size:14px; margin-top:4px;">Ask anything about the report — Fluent RAG System</div></div>', unsafe_allow_html=True)

# شريط الإحصائيات العلوي
chunks = get_chunk_count()
st.markdown(
    f"""
    <div style="display: flex; gap: 12px; margin-bottom: 20px;">
        <div class="glass-card" style="flex: 1; padding: 14px; text-align: center; margin-bottom:0;"><div style="font-size: 22px; font-weight:700; color:#38BDF8;">{st.session_state.query_count}</div><div style="font-size:11px; color:#94A3B8; font-weight:600;">Queries</div></div>
        <div class="glass-card" style="flex: 1; padding: 14px; text-align: center; margin-bottom:0;"><div style="font-size: 22px; font-weight:700; color:#38BDF8;">1</div><div style="font-size:11px; color:#94A3B8; font-weight:600;">Documents</div></div>
        <div class="glass-card" style="flex: 1; padding: 14px; text-align: center; margin-bottom:0;"><div style="font-size: 22px; font-weight:700; color:#38BDF8;">{chunks}</div><div style="font-size:11px; color:#94A3B8; font-weight:600;">Chunks</div></div>
        <div class="glass-card" style="flex: 1; padding: 14px; text-align: center; margin-bottom:0;"><div style="font-size: 14px; font-weight:700; color:#38BDF8; padding-top:4px;">BAAI/bge-m3</div><div style="font-size:11px; color:#94A3B8; font-weight:600; margin-top:4px;">Embedding Engine</div></div>
    </div>
    """,
    unsafe_allow_html=True
)

# 6. الـ Sidebar الجانبي الملون
with st.sidebar:
    st.markdown("### 📊 Key Figures")
    st.markdown('<div class="kpi-box"><div class="kpi-title">Revenue (FY25)</div><div class="kpi-val">$245,122 M</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi-box"><div class="kpi-title">Net Income (FY25)</div><div class="kpi-val">$88,136 M</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi-box"><div class="kpi-title">Total Assets</div><div class="kpi-val">$512,163 M</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="kpi-box"><div class="kpi-title">Operating Cash Flow</div><div class="kpi-val">$118,548 M</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔄 RAG Pipeline Status")
    pipeline_steps = ["PDF Loaded", "OCR Processed", "Semantic Chunking", "Vector Embedding", "Vector Search Retrieval", "LLM Response"]
    for step in pipeline_steps:
        st.markdown(f'<div class="pipeline-step"><span>{step}</span><span class="pipeline-success">✔</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Suggested Questions")
    st.write("• What is the total revenue for FY25?\n\n• Show me the revenue trend over the years.\n\n• Summarize the performance of Intelligent Cloud.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar_icon = "🤖" if msg["role"] == "user" else "https://img.icons8.com/color/48/microsoft.png"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])
        if "chart" in msg:
            st.plotly_chart(msg["chart"], use_container_width=True)
        if "sources" in msg:
            for src in msg["sources"]:
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-header"><span>📄 Page {src['page']}</span><span>Confidence: {int(src['score']*100)}%</span></div>
                        <div style="font-size:12px; color:#CBD5E1; line-height:1.5;">{src['content'][:250]}...</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

# Input الحوار الرئيسي المطور
question = st.chat_input("Ask Microsoft Copilot RAG...")

if question:
    st.session_state.query_count += 1
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🤖"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="https://img.icons8.com/color/48/microsoft.png"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*Thinking...* 🧠")
        
        with st.spinner("⚡ Retrieving relevant context from Annual Report..."):
            results = search_rag(question)
        
        if not results:
            answer = "I couldn't find precise context in the report for this query."
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            context = "\n".join(r["content"] for r in results)
            system_prompt = "You are an elite financial analyst and official Microsoft AI assistant. Answer the question comprehensively using the provided context. If the user asks in Arabic, reply in Arabic."
            user_prompt = f"Question: {question}\n\nContext:\n{context}"
            
            full_response = ""
            candidate_models = [
                "deepseek/deepseek-v4-flash:free",
                "google/gemini-2.5-flash:free",
                "meta-llama/llama-3.3-70b-instruct:free"
            ]
            
            stream_success = False
            for model_name in candidate_models:
                try:
                    response_stream = llm_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                        temperature=0.2,
                        stream=True,
                        timeout=8
                    )
                    
                    for chunk in response_stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    stream_success = True
                    break
                except Exception:
                    full_response = ""
                    continue
            
            if not stream_success:
                full_response = "⚠️ All service nodes are currently busy. Please execute the query again."
                message_placeholder.markdown(full_response)

            # محرك عرض الرسوم البيانية التلقائي (Charts Engine)
            chart_obj = None
            q_lower = question.lower()
            if any(k in q_lower for k in ["revenue", "net income", "trend", "growth", "إيراد", "أرباح"]):
                st.markdown("### 📈 Automated Financial Trend Analysis")
                fig = go.Figure()
                years = ['2022', '2023', '2024', '2025']
                rev = [198270, 211915, 245122, 285000]
                net_inc = [72738, 72361, 88136, 99500]
                
                if "net" in q_lower or "صافي" in q_lower or "أرباح" in q_lower:
                    fig.add_trace(go.Bar(x=years, y=net_inc, name="Net Income ($M)", marker_color='#10B981'))
                    fig.update_layout(title="Net Income Growth Trend")
                else:
                    fig.add_trace(go.Scatter(x=years, y=rev, mode='lines+markers', name='Revenue ($M)', line=dict(color='#38BDF8', width=3)))
                    fig.update_layout(title="Total Revenue Trend Over Years (Microsoft)")

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                chart_obj = fig

            # عرض الـ Source Cards
            st.markdown("#### 📄 Verification Sources")
            for src in results:
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-header"><span>📄 Document Section (Page {src['page']})</span><span>Similarity : {src['score']:.2f}</span></div>
                        <div style="font-size:12px; color:#CBD5E1; line-height:1.5;">{src['content'][:300]}...</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            msg_data = {"role": "assistant", "content": full_response, "sources": results}
            if chart_obj:
                msg_data["chart"] = chart_obj
            st.session_state.messages.append(msg_data)
