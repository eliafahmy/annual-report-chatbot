"""
Microsoft Annual Report AI — واجهة Streamlit احترافية
=========================================================
Glassmorphism + Fluent-inspired design + KPI cards + مصادر اختيارية
+ رسم بياني تلقائي عند سؤال عن الإيرادات عبر السنين + تأثير كتابة حي
"""

import json
import re
import time

import plotly.graph_objects as go
import requests
import streamlit as st
from huggingface_hub import InferenceClient
from openai import OpenAI
from qdrant_client import QdrantClient

# ---------------------------------------------------------------------------
# إعدادات عامة
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Microsoft Annual Report AI", page_icon="📊", layout="wide")

COLLECTION_NAME = "annual_report_chunks"
EMBED_MODEL = "BAAI/bge-m3"

# ---------------------------------------------------------------------------
# التصميم — Glassmorphism بلمسة Fluent Design (خلفية متدرجة متحركة + زجاج)
# ملحوظة: مفيش أي شعار خاص بمايكروسوفت هنا (ده IP محمي)، الألوان بس
# مستوحاة من هوية Fluent العامة (أزرق تقني هادئ).
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Tajawal:wght@400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Tajawal', 'Poppins', sans-serif; }

    .stApp {
        background: linear-gradient(-45deg, #0B1D3A, #123A5E, #0E2A4A, #163F63);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
    }
    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .glass {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }

    .hero {
        padding: 30px 34px;
        margin-bottom: 22px;
        text-align: center;
    }
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 32px;
        background: linear-gradient(90deg, #7FD4FF, #4FA6FF, #A98CFF);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        color: #C9D9EE;
        font-size: 15px;
        margin-top: 8px;
    }

    .kpi-card {
        padding: 16px 18px;
        margin-bottom: 14px;
        color: #F2F6FC;
    }
    .kpi-label {
        font-size: 12px;
        color: #9FC0E8;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 20px;
        font-weight: 700;
        color: #FFFFFF;
    }
    .kpi-check { color: #52D68B; margin-left: 6px; }

    [data-testid="stChatMessage"] { background: transparent !important; padding: 4px 0 !important; }
    [data-testid="stChatMessageContent"] {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 16px;
        padding: 14px 18px;
        color: #F2F6FC !important;
    }
    [data-testid="stChatMessageContent"] p { color: #F2F6FC !important; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(79,166,255,0.25), rgba(169,140,255,0.20));
        border: 1px solid rgba(127,212,255,0.35);
    }

    [data-testid="stChatInput"] {
        border: 1.5px solid rgba(127,212,255,0.5) !important;
        border-radius: 16px !important;
        background: rgba(255,255,255,0.07) !important;
    }
    [data-testid="stChatInput"] textarea { color: #F2F6FC !important; }

    .source-card {
        padding: 12px 16px;
        margin-bottom: 10px;
        color: #E8F0FB;
        font-size: 13px;
    }
    .source-card .src-title { color: #7FD4FF; font-weight: 600; margin-bottom: 4px; }
    .source-card .src-meta { color: #9FC0E8; font-size: 12px; margin-bottom: 6px; }

    .stButton button {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(127,212,255,0.4);
        color: #E8F0FB;
        border-radius: 12px;
    }
    .stButton button:hover {
        background: rgba(127,212,255,0.2);
        border-color: #7FD4FF;
    }

    section[data-testid="stSidebar"] { background: rgba(6, 20, 40, 0.55); backdrop-filter: blur(10px); }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="glass hero">
        <div class="hero-title">📊 Microsoft Annual Report AI</div>
        <div class="hero-subtitle">Ask anything about the report — اسأل بأي لغة تحبها</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# الاتصالات
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_hf_client():
    return InferenceClient(api_key=st.secrets["HF_TOKEN"])


@st.cache_resource(show_spinner=False)
def load_qdrant_client():
    return QdrantClient(url=st.secrets["qdranturl"], api_key=st.secrets["qdrantapi"])


@st.cache_resource(show_spinner=False)
def load_llm_client():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])


hf_client = load_hf_client()
qdrant = load_qdrant_client()
llm_client = load_llm_client()


@st.cache_data(ttl=3600, show_spinner=False)
def get_live_free_models(limit=5):
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        data = resp.json()["data"]
        free = [
            m["id"] for m in data
            if float(m.get("pricing", {}).get("prompt", "1")) == 0.0
            and float(m.get("pricing", {}).get("completion", "1")) == 0.0
        ]
        return free[:limit]
    except Exception:
        return ["deepseek/deepseek-v4-flash:free"]


# ---------------------------------------------------------------------------
# البحث
# ---------------------------------------------------------------------------
def embed_query(text):
    vec = hf_client.feature_extraction(text, model=EMBED_MODEL)
    try:
        import numpy as np
        arr = np.array(vec)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        return arr.tolist()
    except Exception:
        return list(vec)


def search(query, top_k=6):
    q_dense = embed_query(query)
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_dense,
        using="dense",
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "content": r.payload["content"],
            "header_path": r.payload.get("header_path") or "بدون عنوان",
            "page": r.payload.get("page"),
            "score": float(r.score),
        }
        for r in results
    ]


def build_context(results):
    return "\n\n---\n\n".join(r["content"] for r in results)


def call_llm(system_prompt, user_prompt, temperature=0.2):
    fallback_models = get_live_free_models()
    for model_name in fallback_models:
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                timeout=30,
            )
            return response.choices[0].message.content
        except Exception:
            continue
    return None


ANSWER_SYSTEM_PROMPT = """You are a precise financial-report assistant.
Answer strictly using the provided context below — never invent numbers or facts not present in it.
If the answer isn't in the context, say clearly that the information isn't available in the retrieved sections.

Language rule (very important): detect the language the user asked in (Arabic, English, or any other
language) and answer FULLY and NATURALLY in that same language, regardless of the source document's language.

Do not mention sources, references, page numbers, or bracket citations like [1] in your answer text itself.
Give a clean, direct, well-written answer as if you already know the report by heart."""


def ask(question, top_k=6):
    results = search(question, top_k=top_k)
    if not results:
        return "معلش، مش لاقي معلومة مرتبطة بالسؤال ده في التقرير.", []

    context = build_context(results)
    user_prompt = f"Question: {question}\n\nContext:\n{context}"
    answer = call_llm(ANSWER_SYSTEM_PROMPT, user_prompt)
    if answer is None:
        answer = "⚠️ في مشكلة مؤقتة في الوصول للموديل، جرب تاني بعد شوية."
    return answer, results


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


# ---------------------------------------------------------------------------
# KPI cards — بتتحسب مرة واحدة بس لكل جلسة
# ---------------------------------------------------------------------------
KPI_SYSTEM_PROMPT = """Extract the requested financial figures from the context.
Respond with ONLY a compact JSON object, no extra text, in this exact shape:
{"revenue": "value or null", "net_income": "value or null", "total_assets": "value or null", "operating_cash_flow": "value or null"}
Use the same currency/units shown in the context (e.g. "$281,724M"). If a figure isn't in the context, use null."""


@st.cache_data(ttl=3600, show_spinner=False)
def compute_kpis():
    try:
        results = search(
            "total revenue, net income, total assets, and cash flow from operations for the most recent fiscal year",
            top_k=8,
        )
        context = build_context(results)
        raw = call_llm(KPI_SYSTEM_PROMPT, f"Context:\n{context}", temperature=0.0)
        data = extract_json_block(raw) or {}
        return data
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# الشارت — بيظهر بس لو السؤال عن اتجاه الإيرادات عبر السنين
# ---------------------------------------------------------------------------
CHART_TRIGGERS = [
    "trend", "over the years", "by year", "yearly", "historical", "over time",
    "عبر السنين", "على مدار السنين", "بالسنوات", "اتجاه الإيرادات", "خلال السنوات",
]

CHART_SYSTEM_PROMPT = """Extract a yearly revenue series from the context if present.
Respond with ONLY a compact JSON object, no extra text, in this exact shape:
{"years": ["2023", "2024", "2025"], "values": [211915, 245122, 281724]}
Values must be plain numbers (no currency symbols or commas). If you can't find a multi-year series, respond with {"years": [], "values": []}."""


def maybe_render_chart(question, results):
    q_lower = question.lower()
    if "revenue" not in q_lower and "إيراد" not in question:
        return
    if not any(t in q_lower or t in question for t in CHART_TRIGGERS):
        return
    try:
        context = build_context(results)
        raw = call_llm(CHART_SYSTEM_PROMPT, f"Context:\n{context}", temperature=0.0)
        data = extract_json_block(raw)
        if not data or not data.get("years") or not data.get("values"):
            return
        fig = go.Figure(
            go.Bar(
                x=data["years"], y=data["values"],
                marker=dict(color="#4FA6FF"),
                text=data["values"], textposition="outside",
            )
        )
        fig.update_layout(
            title="Revenue by Year",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E8F0FB"),
            height=320,
            margin=dict(t=40, b=20, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        return


# ---------------------------------------------------------------------------
# تأثير الكتابة الحية
# ---------------------------------------------------------------------------
def type_effect(text, chunk_size=4, delay=0.015):
    words = text.split(" ")
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size]) + " "
        time.sleep(delay)


# ---------------------------------------------------------------------------
# الشريط الجانبي — KPI Cards + مفتاح عرض المصادر
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📈 Key Figures")
    kpis = compute_kpis()
    labels = {
        "revenue": "Revenue", "net_income": "Net Income",
        "total_assets": "Total Assets", "operating_cash_flow": "Cash Flow",
    }
    for key, label in labels.items():
        value = kpis.get(key) if kpis else None
        display_value = value if value else "—"
        st.markdown(
            f"""
            <div class="glass kpi-card">
                <div class="kpi-label">{label} <span class="kpi-check">✔</span></div>
                <div class="kpi-value">{display_value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    show_sources = st.toggle("📄 اعرض المصادر", value=False)

    st.markdown("---")
    st.markdown("### 💡 إيه اللي ممكن تسأله؟")
    help_categories = [
        ("💰", "الأرقام المالية", [
            "إجمالي الإيرادات كام؟",
            "What was the net income this year?",
        ]),
        ("📈", "الاتجاهات والنمو", [
            "Show revenue trend over the years",
            "إزاي نمت الإيرادات مقارنة بالسنة اللي فاتت؟",
        ]),
        ("🏢", "قطاعات العمل", [
            "What are the main business segments?",
            "إيه أكبر قطاع من حيث الإيرادات؟",
        ]),
        ("⚠️", "المخاطر والتحديات", [
            "أهم المخاطر المذكورة في التقرير إيه؟",
            "What risks does the company highlight?",
        ]),
    ]
    for icon, title, examples in help_categories:
        with st.expander(f"{icon} {title}"):
            for ex in examples:
                st.markdown(f"- {ex}")


# ---------------------------------------------------------------------------
# الشات
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

clicked = None
if not st.session_state.messages:
    st.markdown('<p style="color:#C9D9EE; font-size:13px;">جرب تسأل:</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    suggestions = [
        "إجمالي الإيرادات كام؟",
        "What are the main business segments?",
        "أهم المخاطر المذكورة إيه؟",
    ]
    for col, s in zip(cols, suggestions):
        if col.button(s, use_container_width=True):
            clicked = s

for msg in st.session_state.messages:
    avatar = "🧑‍💼" if msg["role"] == "user" else "📊"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources") and show_sources:
            for i, r in enumerate(msg["sources"], start=1):
                page_info = f"Page {r['page']}" if r.get("page") is not None else "Page not available"
                st.markdown(
                    f"""
                    <div class="glass source-card">
                        <div class="src-title">📄 Source [{i}] — {r['header_path']}</div>
                        <div class="src-meta">{page_info} · Similarity: {r['score']:.2f}</div>
                        {r['content'][:200]}...
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

question = st.chat_input("اكتب سؤالك هنا بأي لغة...") or clicked

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("بدوّر في التقرير..."):
            answer, sources = ask(question)
        st.write_stream(type_effect(answer))
        maybe_render_chart(question, sources)
        if sources and show_sources:
            for i, r in enumerate(sources, start=1):
                page_info = f"Page {r['page']}" if r.get("page") is not None else "Page not available"
                st.markdown(
                    f"""
                    <div class="glass source-card">
                        <div class="src-title">📄 Source [{i}] — {r['header_path']}</div>
                        <div class="src-meta">{page_info} · Similarity: {r['score']:.2f}</div>
                        {r['content'][:200]}...
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
