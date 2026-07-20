"""
شات بوت RAG على التقرير السنوي — واجهة Streamlit (نسخة خفيفة)
================================================================
البايبلاين: bge-m3 embeddings عن طريق Hugging Face Inference API (مش محلي)
           -> Qdrant (Dense Search) -> BGE Reranker عن طريق HF Inference API (اختياري)
           -> موديل نصي مجاني على OpenRouter -> إجابة + مصادر

ملحوظة: النسخة دي مفيهاش أي موديلات تتحمّل محليًا (لا torch ولا transformers)،
عشان تشتغل مرتاح جدًا على الموارد المحدودة لـ Streamlit Community Cloud.
"""

import requests
import streamlit as st
from qdrant_client import QdrantClient, models
from huggingface_hub import InferenceClient
from openai import OpenAI

# ---------------------------------------------------------------------------
# إعدادات عامة
# ---------------------------------------------------------------------------
st.set_page_config(page_title="مساعد التقرير السنوي", page_icon="📊", layout="centered")

COLLECTION_NAME = "annual_report_chunks"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


# ---------------------------------------------------------------------------
# الاتصالات (بتتعمل مرة واحدة بس وتفضل محفوظة في الكاش)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_hf_client():
    return InferenceClient(api_key=st.secrets["HF_TOKEN"])


@st.cache_resource(show_spinner="جاري الاتصال بـ Qdrant...")
def load_qdrant_client():
    return QdrantClient(url=st.secrets["qdranturl"], api_key=st.secrets["qdrantapi"])


@st.cache_resource(show_spinner=False)
def load_llm_client():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])


hf_client = load_hf_client()
qdrant = load_qdrant_client()
llm_client = load_llm_client()


@st.cache_data(ttl=3600, show_spinner=False)
def get_live_free_models(limit=8):
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        data = resp.json()["data"]
        free = [
            m["id"]
            for m in data
            if float(m.get("pricing", {}).get("prompt", "1")) == 0.0
            and float(m.get("pricing", {}).get("completion", "1")) == 0.0
        ]
        return free[:limit]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# البحث: embedding عن طريق HF Inference API + بحث Dense في Qdrant + reranking اختياري
# ---------------------------------------------------------------------------
def embed_query(text):
    """يجيب dense embedding لسؤال المستخدم عن طريق HF Inference API (نفس موديل bge-m3
    اللي استخدمناه في رفع البيانات، فالمقارنة صحيحة)."""
    vec = hf_client.feature_extraction(text, model=EMBED_MODEL)
    # ممكن يرجع list of lists (لو الموديل رجّع per-token) أو vector واحد، بناخد أول صف لو الحالة الأولى
    try:
        import numpy as np
        arr = np.array(vec)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)  # احتياط: لو رجع per-token، ناخد المتوسط
        return arr.tolist()
    except Exception:
        return list(vec)


def rerank(query, results):
    """يحاول يعمل rerank عن طريق HF Inference API. لو فشل (الموديل مش متاح
    على الـ serverless API دلوقتي)، بيرجّع نفس الترتيب الأصلي من غير ما يوقف التطبيق."""
    try:
        scored = []
        for r in results:
            out = hf_client.text_classification(
                text=query,
                text_pair=r.payload["content"][:2000],
                model=RERANK_MODEL,
            )
            score = out[0].score if isinstance(out, list) else out.score
            scored.append((r, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    except Exception:
        # فشل الـ reranker؟ نكمل بترتيب الـ Qdrant الأصلي (score بتاعه هو الأساس)
        return [(r, r.score) for r in results]


def search(query, top_k=5, candidates=15):
    q_dense = embed_query(query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_dense,
        using="dense",
        limit=candidates,
        with_payload=True,
    ).points

    if not results:
        return []

    reranked = rerank(query, results)[:top_k]
    return [
        {
            "score": float(score),
            "content": r.payload["content"],
            "header_path": r.payload["header_path"],
            "type": r.payload["type"],
            "page": r.payload["page"],
        }
        for r, score in reranked
    ]


# ---------------------------------------------------------------------------
# توليد الإجابة
# ---------------------------------------------------------------------------
def build_context(results):
    blocks = []
    for i, r in enumerate(results, start=1):
        page_info = f"صفحة {r['page']}" if r.get("page") is not None else "رقم الصفحة غير متاح"
        blocks.append(
            f"[{i}] (المصدر: {r['header_path']} | {page_info} | نوع المحتوى: {r['type']})\n{r['content']}"
        )
    return "\n\n---\n\n".join(blocks)


SYSTEM_PROMPT = """أنت مساعد ذكي متخصص في تحليل التقارير المالية والسنوية.
مهمتك: تجاوب على سؤال المستخدم بالاعتماد فقط على المقاطع المرفقة تحت (Context)، من غير ما تخترع أي معلومة مش موجودة فيها.

قواعد صارمة:
- لو الإجابة مش موجودة في الـ Context المرفق، قول بوضوح إن المعلومة غير متوفرة في المقاطع المسترجَعة.
- لما تستخدم معلومة من مقطع معين، اذكر رقمه المرجعي بين قوسين هكذا: [1], [2].
- جاوب باللغة اللي اتسأل بيها السؤال (عربي أو إنجليزي).
- لو الإجابة رقم مالي، اذكره بالظبط زي ما هو مكتوب في المصدر من غير تقريب."""


def ask(question, top_k=5):
    results = search(question, top_k=top_k)
    if not results:
        return "معلش، مش لاقي أي مقطع مرتبط بالسؤال ده في التقرير.", []

    user_prompt = f"### السؤال:\n{question}\n\n### المقاطع المسترجَعة (Context):\n{build_context(results)}"

    fallback_models = get_live_free_models()
    if not fallback_models:
        fallback_models = ["deepseek/deepseek-v4-flash:free"]

    response = None
    last_error = None
    for model_name in fallback_models:
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            break
        except Exception as e:
            last_error = e
            continue

    if response is None:
        return f"⚠️ كل الموديلات المجانية فشلت، آخر خطأ: {last_error}", results

    return response.choices[0].message.content, results


# ---------------------------------------------------------------------------
# الواجهة
# ---------------------------------------------------------------------------
st.title("📊 مساعد التقرير السنوي")
st.caption("اسأل أي سؤال عن التقرير السنوي، والإجابة هتيجي مبنية على النص الفعلي مع ذكر المصادر.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 المصادر المستخدمة"):
                for i, r in enumerate(msg["sources"], start=1):
                    page_info = f"صفحة {r['page']}" if r.get("page") is not None else "رقم الصفحة غير متاح"
                    st.markdown(f"**[{i}]** {r['header_path']} — {page_info} (score: {r['score']:.3f})")
                    st.text(r["content"][:300] + ("..." if len(r["content"]) > 300 else ""))

question = st.chat_input("اكتب سؤالك عن التقرير هنا...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في التقرير وتوليد الإجابة..."):
            answer, sources = ask(question)
        st.markdown(answer)
        if sources:
            with st.expander("📎 المصادر المستخدمة"):
                for i, r in enumerate(sources, start=1):
                    page_info = f"صفحة {r['page']}" if r.get("page") is not None else "رقم الصفحة غير متاح"
                    st.markdown(f"**[{i}]** {r['header_path']} — {page_info} (score: {r['score']:.3f})")
                    st.text(r["content"][:300] + ("..." if len(r["content"]) > 300 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
