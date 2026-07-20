"""
مساعد التقرير السنوي — تصميم "دفتر الأستاذ" مع خلفية متحركة
==========================================================================
✔ خلفية جلد مع أنيميشن حبر متحرك + جزيئات ذهبية طافية
✔ ورقة دفتر مع سطور وهامش أحمر + تجليد نحاسي
✔ لوحة رأسية محفورة بشريط نحاسي متلألئ + ختم شمعي
✔ إصلاح: embed 3D / qdrant fallback / أخطاء الاتصال
"""

import requests
import numpy as np
import streamlit as st
from qdrant_client import QdrantClient
from huggingface_hub import InferenceClient
from openai import OpenAI

# ---------------------------------------------------------------------------
# إعدادات عامة
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="مساعد التقرير السنوي",
    page_icon="📑",
    layout="centered",
)

COLLECTION_NAME = "annual_report_chunks"
EMBED_MODEL = "BAAI/bge-m3"

# ---------------------------------------------------------------------------
# التصميم — دفتر الأستاذ + أنيميشن
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Tajawal:wght@400;500;700&family=Inter:wght@400;500&display=swap');

    :root {
        --leather:       #1A1410;
        --leather-mid:   #2A1F17;
        --leather-light: #3D2E22;
        --brass:         #B8862E;
        --brass-light:   #D9A94A;
        --brass-shine:   #E8C068;
        --paper:         #FAF6EC;
        --paper-warm:    #F3ECDA;
        --paper-aged:    #EDE4CC;
        --ink:           #0E2A2B;
        --ink-light:     #1C3D3E;
        --text:          #22201B;
        --muted:         #6E6857;
        --stamp-red:     #8B2500;
    }

    html, body, [class*="css"] {
        font-family: 'Tajawal', 'Inter', sans-serif;
    }

    /* ═══════════════════════════════════════════════════════
       خلفية الجلد — مع أنيميشن "حبر ينتشر"
       ═══════════════════════════════════════════════════════ */
    .stApp {
        background: var(--leather) !important;
        position: relative;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 600px 400px at 15% 25%, rgba(184,134,46,0.09), transparent),
            radial-gradient(ellipse 500px 500px at 85% 75%, rgba(184,134,46,0.07), transparent),
            radial-gradient(ellipse 700px 300px at 50% 10%, rgba(139,37,0,0.04), transparent),
            radial-gradient(ellipse 400px 600px at 70% 40%, rgba(232,192,104,0.05), transparent),
            radial-gradient(ellipse 300px 500px at 25% 80%, rgba(184,134,46,0.06), transparent);
        background-size: 200% 200%;
        animation: inkSpread 22s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    /* ملمس الجلد الدقيق */
    .stApp::after {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            repeating-linear-gradient(0deg,   rgba(184,134,46,0.018) 0px, transparent 1px, transparent 3px),
            repeating-linear-gradient(90deg,  rgba(184,134,46,0.012) 0px, transparent 1px, transparent 4px),
            repeating-linear-gradient(45deg,  rgba(184,134,46,0.008) 0px, transparent 1px, transparent 6px);
        pointer-events: none;
        z-index: 0;
    }

    @keyframes inkSpread {
        0%   { background-position: 0% 0%; }
        25%  { background-position: 40% 30%; }
        50%  { background-position: 80% 60%; }
        75%  { background-position: 30% 70%; }
        100% { background-position: 0% 0%; }
    }

    /* ═══════════════════════════════════════════════════════
       جزيئات الحبر الذهبية الطافية
       ═══════════════════════════════════════════════════════ */
    .ink-particles {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .ink-p {
        position: absolute;
        border-radius: 50%;
        opacity: 0;
        animation: inkRise linear infinite;
        filter: blur(0.5px);
    }
    @keyframes inkRise {
        0%   { transform: translateY(105vh) scale(0.3) rotate(0deg);   opacity: 0;   }
        12%  { opacity: 0.7; }
        80%  { opacity: 0.3; }
        100% { transform: translateY(-8vh)  scale(1)   rotate(200deg); opacity: 0;   }
    }

    /* ═══════════════════════════════════════════════════════
       تجليد الدفتر — شريط عمودي على اليسار
       ═══════════════════════════════════════════════════════ */
    .ledger-spine {
        position: fixed;
        left: 0;
        top: 0;
        bottom: 0;
        width: 18px;
        background: linear-gradient(90deg, var(--leather-mid), var(--leather-light) 60%, var(--brass) 85%, var(--brass-light) 95%, var(--brass) 100%);
        z-index: 2;
        box-shadow: 2px 0 8px rgba(0,0,0,0.4);
    }
    .ledger-spine::after {
        content: "";
        position: absolute;
        top: 0; bottom: 0;
        left: 9px;
        width: 2px;
        background: repeating-linear-gradient(
            to bottom,
            transparent 0px, transparent 14px,
            var(--paper) 14px, var(--paper) 16px,
            transparent 16px, transparent 30px
        );
        opacity: 0.35;
    }

    /* ═══════════════════════════════════════════════════════
       رأس الصفحة — لوحة نحاسية محفورة
       ═══════════════════════════════════════════════════════ */
    .ledger-header {
        background: linear-gradient(145deg, var(--ink) 0%, var(--ink-light) 60%, var(--ink) 100%);
        border-radius: 16px;
        padding: 22px 26px 22px 78px;
        margin: 0 0 20px 0;
        border: 2px solid var(--brass);
        position: relative;
        overflow: hidden;
        box-shadow:
            0 6px 24px rgba(0,0,0,0.4),
            inset 0 1px 0 rgba(217,169,74,0.15),
            inset 0 -1px 0 rgba(0,0,0,0.3);
        z-index: 1;
    }

    /* شريط نحاسي علوي متلألئ */
    .ledger-header::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg,
            transparent 5%, var(--brass) 15%, var(--brass-shine) 35%,
            var(--brass) 50%, var(--brass-shine) 65%, var(--brass) 85%, transparent 95%);
        background-size: 200% 100%;
        animation: brassSlide 4s linear infinite;
    }
    .ledger-header::after {
        content: "";
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg,
            transparent 5%, var(--brass) 15%, var(--brass-shine) 35%,
            var(--brass) 50%, var(--brass-shine) 65%, var(--brass) 85%, transparent 95%);
        background-size: 200% 100%;
        animation: brassSlide 4s linear infinite reverse;
    }

    @keyframes brassSlide {
        0%   { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* ختم شمعي نابض */
    .ledger-seal {
        position: absolute;
        top: 50%;
        left: 18px;
        transform: translateY(-50%);
        width: 46px;
        height: 46px;
        border: 2.5px solid var(--brass);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        background: radial-gradient(circle, rgba(184,134,46,0.18), transparent 70%);
        box-shadow: 0 0 14px rgba(184,134,46,0.25);
        animation: sealPulse 4s ease-in-out infinite;
    }
    @keyframes sealPulse {
        0%, 100% { box-shadow: 0 0 10px rgba(184,134,46,0.15); transform: translateY(-50%) scale(1); }
        50%      { box-shadow: 0 0 26px rgba(184,134,46,0.45); transform: translateY(-50%) scale(1.04); }
    }

    .ledger-title {
        font-family: 'Fraunces', serif;
        color: var(--paper);
        font-size: 24px;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 6px rgba(0,0,0,0.4);
    }
    .ledger-subtitle {
        color: var(--brass-light);
        font-size: 13px;
        margin-top: 5px;
        font-weight: 400;
    }

    /* ═══════════════════════════════════════════════════════
       منطقة المحتوى — ورقة الدفتر
       ═══════════════════════════════════════════════════════ */
    .ledger-page {
        position: relative;
        z-index: 1;
        background: var(--paper);
        border-radius: 2px 16px 16px 2px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow:
            -5px 0 0 0 var(--brass),
            -7px 0 0 0 rgba(184,134,46,0.20),
            0 6px 28px rgba(0,0,0,0.35),
            0 2px 6px rgba(0,0,0,0.15);
        overflow: hidden;
    }

    /* سطور أفقية */
    .ledger-page::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image: repeating-linear-gradient(
            to bottom,
            transparent 0px, transparent 27px,
            rgba(14,42,43,0.05) 27px, rgba(14,42,43,0.05) 28px
        );
        pointer-events: none;
        border-radius: inherit;
    }

    /* خط هامش أحمر */
    .ledger-page::after {
        content: "";
        position: absolute;
        top: 0; bottom: 0;
        left: 38px;
        width: 2px;
        background: rgba(139,37,0,0.09);
        pointer-events: none;
    }

    .ledger-page-inner {
        position: relative;
        z-index: 1;
    }

    /* ═══════════════════════════════════════════════════════
       فقاعات الشات
       ═══════════════════════════════════════════════════════ */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 3px 0 !important;
    }

    /* رسائل المساعد — ورقة دفتر */
    [data-testid="stChatMessageContent"] {
        background: var(--paper-warm) !important;
        border-radius: 2px 12px 12px 2px !important;
        padding: 14px 18px !important;
        border-left: 4px solid var(--brass) !important;
        box-shadow: 0 2px 8px rgba(14,42,43,0.08) !important;
        color: var(--text) !important;
        position: relative;
        animation: msgAppear 0.4s ease-out;
    }
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] span {
        color: var(--text) !important;
    }

    @keyframes msgAppear {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* رسائل المستخدم — جلد حبري */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, var(--ink), var(--ink-light)) !important;
        color: var(--paper) !important;
        border-left: 4px solid var(--brass-light) !important;
        box-shadow: 0 2px 10px rgba(14,42,43,0.25) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] span {
        color: var(--paper) !important;
    }

    /* ═══════════════════════════════════════════════════════
       خانة الكتابة
       ═══════════════════════════════════════════════════════ */
    [data-testid="stChatInput"] {
        border: 2px solid var(--brass) !important;
        border-radius: 14px !important;
        background: var(--paper-warm) !important;
        box-shadow: 0 4px 18px rgba(0,0,0,0.22) !important;
        position: relative;
        z-index: 1;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
        font-family: 'Tajawal', sans-serif !important;
    }

    /* ═══════════════════════════════════════════════════════
       الأزرار
       ═══════════════════════════════════════════════════════ */
    .suggestion-label {
        color: var(--brass);
        font-size: 13px;
        margin: 6px 0 8px 2px;
        font-weight: 500;
    }
    .stButton button {
        background: var(--paper-warm) !important;
        border: 1.5px solid var(--brass) !important;
        color: var(--ink) !important;
        border-radius: 10px !important;
        font-size: 13px !important;
        padding: 8px 14px !important;
        font-family: 'Tajawal', sans-serif !important;
        transition: all 0.3s ease !important;
        position: relative;
        z-index: 1;
    }
    .stButton button:hover {
        background: var(--brass) !important;
        color: var(--paper) !important;
        border-color: var(--brass-light) !important;
        box-shadow: 0 0 18px rgba(184,134,46,0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* ═══════════════════════════════════════════════════════
       شريط التمرير
       ═══════════════════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--leather); }
    ::-webkit-scrollbar-thumb {
        background: var(--leather-light);
        border: 1px solid var(--brass);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--brass); }

    /* ═══════════════════════════════════════════════════════
       Spinner
       ═══════════════════════════════════════════════════════ */
    .stSpinner > div > div {
        border-color: var(--brass) transparent transparent transparent !important;
    }
    .stSpinner label, .stSpinner div {
        color: var(--brass-light) !important;
    }

    /* ═══════════════════════════════════════════════════════
       إخفاء
       ═══════════════════════════════════════════════════════ */
    #MainMenu, footer, header { visibility: hidden; }

    /* ═══════════════════════════════════════════════════════
       فاصل زخرفي
       ═══════════════════════════════════════════════════════ */
    .ledger-divider {
        text-align: center;
        margin: 10px 0 14px 0;
        color: var(--brass);
        font-size: 11px;
        letter-spacing: 6px;
        opacity: 0.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# جزيئات الحبر الذهبية الطافية
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="ink-particles">
        <div class="ink-p" style="left:3%;  width:5px; height:5px; background:rgba(184,134,46,0.38); animation-duration:19s; animation-delay:0s;"></div>
        <div class="ink-p" style="left:10%; width:3px; height:3px; background:rgba(217,169,74,0.28); animation-duration:24s; animation-delay:4s;"></div>
        <div class="ink-p" style="left:17%; width:6px; height:6px; background:rgba(232,192,104,0.20); animation-duration:17s; animation-delay:2s;"></div>
        <div class="ink-p" style="left:23%; width:4px; height:4px; background:rgba(184,134,46,0.32); animation-duration:21s; animation-delay:7s;"></div>
        <div class="ink-p" style="left:30%; width:3px; height:3px; background:rgba(217,169,74,0.26); animation-duration:26s; animation-delay:1s;"></div>
        <div class="ink-p" style="left:37%; width:5px; height:5px; background:rgba(184,134,46,0.30); animation-duration:20s; animation-delay:5s;"></div>
        <div class="ink-p" style="left:44%; width:4px; height:4px; background:rgba(232,192,104,0.22); animation-duration:23s; animation-delay:9s;"></div>
        <div class="ink-p" style="left:51%; width:6px; height:6px; background:rgba(184,134,46,0.28); animation-duration:18s; animation-delay:3s;"></div>
        <div class="ink-p" style="left:58%; width:3px; height:3px; background:rgba(217,169,74,0.34); animation-duration:25s; animation-delay:6s;"></div>
        <div class="ink-p" style="left:65%; width:5px; height:5px; background:rgba(184,134,46,0.26); animation-duration:22s; animation-delay:0s;"></div>
        <div class="ink-p" style="left:72%; width:4px; height:4px; background:rgba(232,192,104,0.28); animation-duration:19s; animation-delay:8s;"></div>
        <div class="ink-p" style="left:79%; width:3px; height:3px; background:rgba(184,134,46,0.36); animation-duration:27s; animation-delay:4s;"></div>
        <div class="ink-p" style="left:86%; width:5px; height:5px; background:rgba(217,169,74,0.24); animation-duration:21s; animation-delay:2s;"></div>
        <div class="ink-p" style="left:93%; width:4px; height:4px; background:rgba(184,134,46,0.30); animation-duration:24s; animation-delay:7s;"></div>
        <div class="ink-p" style="left:7%;  width:7px; height:7px; background:rgba(232,192,104,0.16); animation-duration:30s; animation-delay:11s;"></div>
        <div class="ink-p" style="left:48%; width:7px; height:7px; background:rgba(184,134,46,0.18); animation-duration:28s; animation-delay:13s;"></div>
        <div class="ink-p" style="left:76%; width:8px; height:8px; background:rgba(217,169,74,0.14); animation-duration:33s; animation-delay:10s;"></div>
        <div class="ink-p" style="left:35%; width:2px; height:2px; background:rgba(232,192,104,0.40); animation-duration:16s; animation-delay:6s;"></div>
        <div class="ink-p" style="left:60%; width:2px; height:2px; background:rgba(184,134,46,0.42); animation-duration:15s; animation-delay:1s;"></div>
        <div class="ink-p" style="left:90%; width:2px; height:2px; background:rgba(217,169,74,0.38); animation-duration:14s; animation-delay:3s;"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# تجليد الدفتر (شريط عمودي على اليسار)
# ---------------------------------------------------------------------------
st.markdown('<div class="ledger-spine"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# الرأس — لوحة نحاسية محفورة
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="ledger-header">
        <div class="ledger-seal">📑</div>
        <div class="ledger-title">مساعد التقرير السنوي</div>
        <div class="ledger-subtitle">اسأل بأي لغة — الإجابة من نص التقرير الفعلي</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# الاتصالات — مع فحص فوري
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_hf_client():
    try:
        return InferenceClient(api_key=st.secrets["HF_TOKEN"])
    except Exception as e:
        st.error(f"❌ مش قادر أوصِل لـ HuggingFace: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_qdrant_client():
    try:
        client = QdrantClient(url=st.secrets["qdranturl"], api_key=st.secrets["qdrantapi"])
        client.get_collection(COLLECTION_NAME)
        return client
    except Exception as e:
        st.error(f"❌ مش قادر أوصِل لـ Qdrant: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_llm_client():
    try:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
    except Exception as e:
        st.error(f"❌ مش قادر أوصِل لـ OpenRouter: {e}")
        return None


hf_client = load_hf_client()
qdrant = load_qdrant_client()
llm_client = load_llm_client()

if hf_client is None or qdrant is None or llm_client is None:
    st.stop()


# ---------------------------------------------------------------------------
# الموديلز المجانية
# ---------------------------------------------------------------------------
FALLBACK_MODELS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-4-scout:free",
    "qwen/qwen3-32b:free",
]


@st.cache_data(ttl=1800, show_spinner=False)
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
        return free[:limit] if free else FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


# ---------------------------------------------------------------------------
# Embedding — إصلاح 3D / 2D / 1D
# ---------------------------------------------------------------------------
def embed_query(text):
    vec = hf_client.feature_extraction(text, model=EMBED_MODEL)
    arr = np.array(vec)
    if arr.ndim == 3:
        arr = arr[0].mean(axis=0)
    elif arr.ndim == 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.mean(axis=0)
    return arr.tolist()


# ---------------------------------------------------------------------------
# البحث — مع fallback لـ qdrant.search()
# ---------------------------------------------------------------------------
def search(query, top_k=6):
    q_dense = embed_query(query)
    try:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=q_dense,
            using="dense",
            limit=top_k,
            with_payload=True,
        ).points
    except (AttributeError, TypeError):
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=("dense", q_dense),
            limit=top_k,
            with_payload=True,
        )

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


# ---------------------------------------------------------------------------
# الـ LLM
# ---------------------------------------------------------------------------
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

    models = get_live_free_models()
    for model_name in models:
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                timeout=20,
            )
            content = response.choices[0].message.content
            if content:
                return content
        except Exception:
            continue

    return "⚠️ في مشكلة مؤقتة في الوصول للموديل، جرب تاني بعد شوية."


# ---------------------------------------------------------------------------
# الواجهة
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

clicked = None

if not st.session_state.messages:
    st.markdown(
        """
        <div class="ledger-page">
            <div class="ledger-page-inner">
                <div class="suggestion-label">✦ جرب تسأل:</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
    avatar = "🧑‍💼" if msg["role"] == "user" else "📑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if st.session_state.messages:
    st.markdown(
        '<div class="ledger-divider">✦ ✦ ✦</div>',
        unsafe_allow_html=True,
    )

question = st.chat_input("اكتب سؤالك هنا بأي لغة...") or clicked

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📑"):
        try:
            with st.spinner("🖊️ بدوّر في الدفتر..."):
                answer = ask(question)
        except Exception as e:
            answer = f"❌ حصل خطأ: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
