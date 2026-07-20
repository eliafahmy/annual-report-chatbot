"""
شات بوت RAG على التقرير السنوي — واجهة Streamlit
==================================================
البايبلاين: bge-m3 (dense+sparse) -> Qdrant (Hybrid Search) -> CrossEncoder Reranker
           -> موديل نصي مجاني على OpenRouter -> إجابة + مصادر
"""

import json
import requests
import streamlit as st
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
from openai import OpenAI

# ---------------------------------------------------------------------------
# إعدادات عامة
# ---------------------------------------------------------------------------
st.set_page_config(page_title="مساعد التقرير السنوي", page_icon="📊", layout="centered")

COLLECTION_NAME = "annual_report_chunks"


# ---------------------------------------------------------------------------
# تحميل الموديلات والاتصالات (بتتحمل مرة واحدة بس وتفضل محفوظة في الكاش)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="جاري تحميل موديل الـ Embedding (bge-m3)...")
def load_embed_model():
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)


@st.cache_resource(show_spinner="جاري تحميل موديل الـ Reranker...")
def load_reranker():
    return CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024, trust_remote_code=False)


@st.cache_resource(show_spinner="جاري الاتصال بـ Qdrant...")
def load_qdrant_client():
    url = st.secrets["qdranturl"]
    api_key = st.secrets["qdrantapi"]
    return QdrantClient(url=url, api_key=api_key)


@st.cache_resource(show_spinner=False)
def load_llm_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"],
    )


@st.cache_data(ttl=3600, show_spinner=False)
def get_live_free_models(limit=8):
    """يجيب أحدث قايمة موديلات مجانية فعليًا من OpenRouter وقت التشغيل."""
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


embed_model = load_embed_model()
reranker = load_reranker()
qdrant = load_qdrant_client()
llm_client = load_llm_client()


# ---------------------------------------------------------------------------
# دوال البحث والإجابة (نفس منطق النوتبوك)
# ---------------------------------------------------------------------------
def sparse_dict_to_qdrant(sparse_weights):
    indices = [int(k) for k in sparse_weights.keys()]
    values = [float(v) for v in sparse_weights.values()]
    return models.SparseVector(indices=indices, values=values)


def search(query, top_k=5, candidates=20):
    q_out = embed_model.encode([query], return_dense=True, return_sparse=True)
    q_dense = q_out["dense_vecs"][0].tolist()
    q_sparse = sparse_dict_to_qdrant(q_out["lexical_weights"][0])

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=q_dense, using="dense", limit=candidates),
            models.Prefetch(query=q_sparse, using="sparse", limit=candidates),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=candidates,
        with_payload=True,
    ).points

    if not results:
        return []

    pairs = [[query, r.payload["content"]] for r in results]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:top_k]
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
