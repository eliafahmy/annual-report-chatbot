"""
مساعد التقرير السنوي — واجهة Streamlit (تصميم "دفتر الأستاذ" + أداء محسّن)
==========================================================================
"""

import requests
import streamlit as st
from qdrant_client import QdrantClient
from huggingface_hub import InferenceClient
from openai import OpenAI

# ---------------------------------------------------------------------------
# إعدادات عامة
# ---------------------------------------------------------------------------
st.set_page_config(page_title="مساعد التقرير السنوي", page_icon="📑", layout="centered")

COLLECTION_NAME = "annual_report_chunks"
EMBED_MODEL = "BAAI/bge-m3"

# ---------------------------------------------------------------------------
# التصميم — لوحة ألوان "دفتر الأستاذ": حبر كحلي غامق + نحاسي دافئ + ورق عاجي
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Tajawal:wght@400;500;700&family=Inter:wght@400;500&display=swap');

    :root {
        --ink: #0E2A2B;
        --ink-light: #14403F;
        --brass: #B8862E;
        --brass-light: #D9A94A;
        --paper: #FAF6EC;
        --paper-warm: #F3ECDA;
        --text: #22201B;
        --muted: #6E6857;
    }

    html, body, [class*="css"] { font-family: 'Tajawal', 'Inter', sans-serif; }

    .stApp {
        background: var(--paper);
    }

    /* رأس الصفحة */
    .ledger-header {
        background: linear-gradient(135deg, var(--ink) 0%, var(--ink-light) 100%);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 22px;
        border: 1px solid var(--brass);
        position: relative;
        overflow: hidden;
    }
    .ledger-header::after {
        content: "";
        position: absolute;
        top: 0; right: 0; bottom: 0;
        width: 6px;
        background: repeating-linear-gradient(
            to bottom, var(--brass) 0px, var(--brass) 8px, transparent 8px, transparent 16px
        );
    }
    .ledger-title {
        font-family: 'Fraunces', serif;
        color: var(--paper);
        font-size: 28px;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.3px;
    }
    .ledger-subtitle {
        color: var(--brass-light);
        font-size: 14px;
        margin-top: 6px;
        font-weight: 400;
    }

    /* فقاعات الشات */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 4px 0 !important;
    }

    [data-testid="stChatMessageContent"] {
        background: var(--paper-warm);
        border-radius: 14px;
        padding: 14px 18px;
        border-left: 3px solid var(--brass);
        box-shadow: 0 1px 3px rgba(14, 42, 43, 0.08);
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background: var(--ink);
        color: var(--paper) !important;
        border-left: 3px solid var(--brass-light);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p {
        color: var(--paper) !important;
    }

    /* خانة الكتابة */
    [data-testid="stChatInput"] {
        border: 1.5px solid var(--brass) !important;
        border-radius: 14px !important;
        background: var(--paper-warm) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
    }

    /* عنوان أسئلة مقترحة */
    .suggestion-label {
        color: var(--muted);
        font-size: 13px;
        margin: 4px 0 8px 0;
        font-weight: 500;
    }

    .stButton button {
        background: var(--paper-warm);
        border: 1px solid var(--brass);
        color: var(--ink);
        border-radius: 10px;
        font-size: 13px;
        padding: 6px 14px;
    }
    .stButton button:hover {
        background: var(--brass);
        color: var(--paper);
        border-color: var(--brass);
    }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="ledger-header">
        <div class="ledger-title">📑 مساعد التقرير السنوي</div>
        <div class="ledger-subtitle">اسأل بأي لغة تحبها — الإجابة هتيجي من نص التقرير الفعلي</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# الاتصالات (كاش، بتتعمل مرة واحدة بس)
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
            m["id"]
            for m in data
            if float(m.get("pricing", {}).get("prompt", "1")) == 0.0
            and float(m.get("pricing", {}).get("completion", "1")) == 0.0
        ]
        return free[:limit]
    except Exception:
        return ["deepseek/deepseek-v4-flash:free"]


# ---------------------------------------------------------------------------
# البحث — نسخة سريعة: embedding واحد + بحث Dense واحد في Qdrant (من غير
# لوب استدعاءات API للـ reranker، وده كان سبب البطء الرئيسي)
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
            "header_path": r.payload.get("header_path"),
            "page": r.payload.get("page"),
        }
        for r in results
    ]


def build_context(results):
    return "\n\n---\n\n".join(r["content"] for r in results)


SYSTEM_PROMPT = """You are a precise financial-report assistant.
Answer strictly using the provided context below — never invent numbers or facts not present in it.
If the answer isn't in the context, say clearly that the information isn't available in the retrieved sections.

Language rule (very important): detect the language the user asked in (Arabic, English, or any other
language) and answer FULLY and NATURALLY in that same language, translating any figures or labels as
needed — regardless of what language the source document is written in.

Do not mention sources, references, page numbers, or bracket citations like [1] in your answer.
Just give a clean, direct, well-written answer as if you already know the report by heart."""


def ask(question, top_k=6):
    results = search(question, top_k=top_k)
    if not results:
        return "معلش، مش لاقي معلومة مرتبطة بالسؤال ده في التقرير."

    context = build_context(results)
    user_prompt = f"Question: {question}\n\nContext:\n{context}"

    fallback_models = get_live_free_models()

    for model_name in fallback_models:
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                timeout=30,
            )
            return response.choices[0].message.content
        except Exception:
            continue

    return "⚠️ في مشكلة مؤقتة في الوصول للموديل، جرب تاني بعد شوية."


# ---------------------------------------------------------------------------
# الواجهة
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown('<div class="suggestion-label">جرب تسأل:</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    suggestions = [
        "إجمالي الإيرادات كام؟",
        "What are the main business segments?",
        "أهم المخاطر المذكورة إيه؟",
    ]
    clicked = None
    for col, s in zip(cols, suggestions):
        if col.button(s, use_container_width=True):
            clicked = s
else:
    clicked = None

for msg in st.session_state.messages:
    avatar = "🧑‍💼" if msg["role"] == "user" else "📑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

question = st.chat_input("اكتب سؤالك هنا بأي لغة...") or clicked

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📑"):
        with st.spinner("بدوّر في التقرير..."):
            answer = ask(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
