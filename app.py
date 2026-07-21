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
        padding: 12px 14px;
        margin-bottom: 6px;
        background: rgba(255, 255, 255, 0.04);
        border-left: 4px solid #38BDF8;
        border-radius: 6px;
    }
    .kpi-title { font-size: 10.5px; color: #E2E8F0 !important; font-weight: 600; text-transform: uppercase; }
    .kpi-val { font-size: 16px; font-weight: 700; color: #38BDF8 !important; margin-top: 1px; }

    /* عناصر الـ RAG Pipeline الجانبية */
    .pipeline-step {
        padding: 6px 12px;
        margin-bottom: 3px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 6px;
        font-size: 11.5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .pipeline-success { color: #34D399 !important; font-weight: bold; }
    .pipeline-active { color: #38BDF8 !important; font-weight: bold; animation: pipelinePulse 1s infinite; }
    .pipeline-pending { color: #475569 !important; font-weight: bold; }
    @keyframes pipelinePulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }

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
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] section {
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        background: #1e293b !important;
    }

    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] textarea * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-size: 15px !important;
        background: transparent !important;
    }

    /* تغيير لون الـ Placeholder (النص المؤقت المكتوب بالداخل) */
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
    }

    /* وضوح خط نص الإجابات جوه فقاعات الشات (كان باهت) */
    [data-testid="stChatMessageContent"],
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] span {
        color: #F1F5F9 !important;
        font-size: 15.5px !important;
        line-height: 1.75 !important;
        font-family: 'Segoe UI', 'Tajawal', sans-serif !important;
    }

    [data-testid="stSidebar"] h3 { margin: 6px 0 4px 0 !important; font-size: 15px !important; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.8rem !important; }
    [data-testid="stSidebar"] .stMarkdown { margin-bottom: 0 !important; }

    /* عناوين المحتوى الرئيسي (كانت باهتة جدًا على الخلفية الغامقة) */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
    .stApp [data-testid="stMarkdownContainer"] h3,
    .stApp [data-testid="stMarkdownContainer"] h4 {
        color: #E8F0FB !important;
        opacity: 1 !important;
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
        return "—"

@st.cache_data(ttl=3600, show_spinner=False)
def get_live_free_models(limit=6):
    """يجيب قايمة الموديلات المجانية الفعلية من OpenRouter وقت التشغيل، بدل قايمة
    ثابتة ممكن تتقادم وتسبب رسالة 'all service nodes are busy' باستمرار."""
    fallback = [
        "deepseek/deepseek-v4-flash:free",
        "meta-llama/llama-4-scout:free",
        "qwen/qwen3-coder:free",
    ]
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        data = resp.json()["data"]
        free = [
            m["id"] for m in data
            if float(m.get("pricing", {}).get("prompt", "1")) == 0.0
            and float(m.get("pricing", {}).get("completion", "1")) == 0.0
        ]
        return free[:limit] if free else fallback
    except Exception:
        return fallback


def clean_answer(text):
    """شبكة أمان: لو الموديل استخدم markdown bold رغم التعليمات، بنشيل النجوم
    ونسيب النص بس، عشان نتفادى تعارض الـ ** مع النص العربي RTL."""
    if not text:
        return text
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("**", "").replace("__", "")
    return text


def clean_snippet(text, max_len=280):
    """بينضف نص الـ chunk من رموز جداول الـ Markdown الخام (|، --- إلخ) عشان
    يبان كنص عادي مقروء في كارت المصدر بدل جدول مبعثر."""
    if not text:
        return ""
    cleaned = re.sub(r"\*\*", "", text)
    cleaned = re.sub(r"\|", " ", cleaned)
    cleaned = re.sub(r"-{2,}", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned[:max_len]


KPI_EXTRACTION_PROMPT = """Extract the requested financial figures from the context below, using
the EXACT numbers shown in the context — never estimate or invent a number.
Respond with ONLY a compact JSON object, no extra text, in this exact shape:
{"revenue": "value or null", "net_income": "value or null", "total_assets": "value or null", "operating_cash_flow": "value or null"}
Use the same currency/units shown in the context (e.g. "$281,724M"). If a figure truly isn't in the context, use null."""


def call_llm_simple(system_prompt, user_prompt, temperature=0.0):
    for model_name in get_live_free_models():
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                timeout=25,
            )
            return response.choices[0].message.content
        except Exception:
            continue
    return None


def extract_json_block(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def compute_real_kpis():
    """بيستخرج الأرقام المالية الحقيقية من التقرير نفسه (مش أرقام ثابتة مكتوبة يدويًا)."""
    try:
        results = search_rag(
            "total revenue, net income, total assets, and cash flow from operations for the most recent fiscal year",
            top_k=8,
        )
        context = "\n\n---\n\n".join(r["content"] for r in results)
        raw = call_llm_simple(KPI_EXTRACTION_PROMPT, f"Context:\n{context}")
        return extract_json_block(raw) or {}
    except Exception:
        return {}


CHART_EXTRACTION_PROMPT = """Extract a yearly revenue series from the context below if present, using
the EXACT figures shown — never invent or estimate numbers that aren't explicitly in the context.
Respond with ONLY a compact JSON object, no extra text, in this exact shape:
{"years": ["2023", "2024", "2025"], "values": [211915, 245122, 281724]}
Values must be plain numbers (no currency symbols or commas). If you can't find a real multi-year
series in the context, respond with {"years": [], "values": []}."""


@st.cache_data(ttl=3600, show_spinner=False)
def compute_real_revenue_trend():
    try:
        results = search_rag("total revenue by year, multi-year revenue history", top_k=8)
        context = "\n\n---\n\n".join(r["content"] for r in results)
        raw = call_llm_simple(CHART_EXTRACTION_PROMPT, f"Context:\n{context}")
        data = extract_json_block(raw)
        if data and data.get("years") and data.get("values"):
            return data
    except Exception:
        pass
    return None

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

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
    kpis = compute_real_kpis()
    kpi_labels = {
        "revenue": "Revenue", "net_income": "Net Income",
        "total_assets": "Total Assets", "operating_cash_flow": "Operating Cash Flow",
    }
    for key, label in kpi_labels.items():
        value = kpis.get(key) if kpis else None
        display_value = value if value else "غير متاح"
        st.markdown(f'<div class="kpi-box"><div class="kpi-title">{label}</div><div class="kpi-val">{display_value}</div></div>', unsafe_allow_html=True)

    st.markdown("### 🔄 RAG Pipeline Status")
    pipeline_placeholder = st.empty()

    st.markdown("### 💡 Suggested Questions")
    st.write("• What is the total revenue for FY25?\n\n• Show me the revenue trend over the years.\n\n• Summarize the performance of Intelligent Cloud.")

    show_sources = st.toggle("📄 Show Sources", value=True)

    st.markdown(
        '<div style="text-align:center; color:#7FD4FF; font-size:12px; font-weight:600; margin-top:14px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.08);">Built by Elia Fahmy</div>',
        unsafe_allow_html=True,
    )

PIPELINE_STEPS = ["PDF Loaded", "OCR Processed", "Semantic Chunking", "Vector Embedding", "Vector Search Retrieval", "LLM Response"]


def render_pipeline(placeholder, active_stage=None):
    """active_stage: None = كله جاهز (حالة سكون) | 'search' = بيدور دلوقتي |
    'llm' = بيولّد الإجابة دلوقتي | 'done' = خلص السؤال ده"""
    html = ""
    for i, step in enumerate(PIPELINE_STEPS):
        if i < 4:
            state = "done"  # مراحل بناء الـ Pipeline، جاهزة دايمًا
        elif i == 4:  # Vector Search Retrieval
            state = "active" if active_stage == "search" else "done"
        else:  # LLM Response
            state = "active" if active_stage == "llm" else ("done" if active_stage in ("done", None) else "pending")

        if state == "active":
            icon = '<span class="pipeline-active">●</span>'
        elif state == "pending":
            icon = '<span class="pipeline-pending">○</span>'
        else:
            icon = '<span class="pipeline-success">✔</span>'
        html += f'<div class="pipeline-step"><span>{step}</span>{icon}</div>'
    placeholder.markdown(html, unsafe_allow_html=True)


render_pipeline(pipeline_placeholder, active_stage=None)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg_idx, msg in enumerate(st.session_state.messages):
    avatar_icon = "🧑‍💼" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])
        if "chart" in msg:
            st.plotly_chart(msg["chart"], use_container_width=True, key=f"chart_history_{msg_idx}")
        if "sources" in msg and show_sources:
            for src in msg["sources"]:
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-header"><span>📄 Page {src['page']}</span><span>Confidence: {int(src['score']*100)}%</span></div>
                        <div style="font-size:12px; color:#CBD5E1; line-height:1.5;">{clean_snippet(src['content'])}...</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

# Input الحوار الرئيسي المطور
question = st.chat_input("اسأل عن التقرير السنوي...")

if question:
    st.session_state.query_count += 1
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*Thinking...* 🧠")

        render_pipeline(pipeline_placeholder, active_stage="search")
        with st.spinner("⚡ Retrieving relevant context from Annual Report..."):
            results = search_rag(question)

        if not results:
            render_pipeline(pipeline_placeholder, active_stage="done")
            answer = "I couldn't find precise context in the report for this query."
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            render_pipeline(pipeline_placeholder, active_stage="llm")
            context = "\n".join(r["content"] for r in results)
            system_prompt = "You are a precise financial-report analysis assistant. Answer strictly using the provided context — never invent numbers or facts not present in it. If the user asks in Arabic, reply in Arabic; otherwise match the question's language. IMPORTANT FORMATTING RULE: write in plain prose only — do NOT use markdown formatting like **bold**, bullet points with asterisks, or headers. Use plain sentences and, if a list is needed, use plain numbered lines (1., 2., 3.) without any asterisks or symbols."
            user_prompt = f"Question: {question}\n\nContext:\n{context}"
            
            full_response = ""
            candidate_models = get_live_free_models()

            stream_success = False
            for model_name in candidate_models:
                try:
                    response_stream = llm_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                        temperature=0.2,
                        stream=True,
                        timeout=20
                    )
                    
                    for chunk in response_stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(clean_answer(full_response) + "▌")
                    full_response = clean_answer(full_response)
                    message_placeholder.markdown(full_response)
                    stream_success = True
                    break
                except Exception:
                    full_response = ""
                    continue
            
            if not stream_success:
                full_response = "⚠️ All service nodes are currently busy. Please execute the query again."
                message_placeholder.markdown(full_response)

            render_pipeline(pipeline_placeholder, active_stage="done")

            # محرك عرض الرسوم البيانية التلقائي (Charts Engine) — بيانات حقيقية بس
            chart_obj = None
            q_lower = question.lower()
            if any(k in q_lower for k in ["revenue", "trend", "growth", "over the years", "إيراد", "عبر السنين", "بالسنوات"]):
                trend_data = compute_real_revenue_trend()
                if trend_data:
                    st.markdown("### 📈 Automated Financial Trend Analysis")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend_data["years"], y=trend_data["values"],
                        mode='lines+markers', name='Revenue ($M)',
                        line=dict(color='#38BDF8', width=3),
                    ))
                    fig.update_layout(
                        title=dict(text="Total Revenue Trend Over Years (من التقرير الفعلي)", font=dict(color="#F2F6FC", size=16)),
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="#E8F0FB"),
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_live_{st.session_state.query_count}")
                    chart_obj = fig
                # لو مفيش سلسلة سنوات حقيقية واضحة في التقرير، مبنعرضش شارت مُلفَّق —
                # أحسن نسكت من إننا نوري رقم غلط.

            # عرض الـ Source Cards
            if show_sources:
                st.markdown("#### 📄 Verification Sources")
                for src in results:
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-header"><span>📄 Document Section (Page {src['page']})</span><span>Similarity : {src['score']:.2f}</span></div>
                            <div style="font-size:12px; color:#CBD5E1; line-height:1.5;">{clean_snippet(src['content'])}...</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            msg_data = {"role": "assistant", "content": full_response, "sources": results}
            if chart_obj:
                msg_data["chart"] = chart_obj
            st.session_state.messages.append(msg_data)

    st.rerun()

