import streamlit as st

from main import build_vectorstore, retrieve_documents, trim_context
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(
    page_title="Chatbot RAG - IA Générative",
    page_icon="🤖",
    layout="centered",
)

st.markdown(
    """
    <style>
      .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
      }
      .stChatMessage {
        border-radius: 16px;
        padding: 4px 2px;
      }
      .rag-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background: rgba(0,0,0,0.06);
        font-size: 0.85rem;
        margin-bottom: 0.75rem;
      }
      .sources a {
        text-decoration: none;
      }
      .sources a:hover {
        text-decoration: underline;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="rag-badge">RAG OpenAI • Wikipedia • Chroma • Streaming</div>',
    unsafe_allow_html=True,
)
st.title("🤖 Chatbot RAG – IA Générative")


with st.sidebar:
    st.header("Réglages")
    k = st.slider("Nombre de chunks (k)", min_value=2, max_value=10, value=5)
    max_chars = st.slider(
        "Taille max du contexte",
        min_value=1000,
        max_value=6000,
        value=4000,
        step=200,
    )
    st.caption("Astuce : si la réponse est trop longue ou trop lente, baisse k ou max_chars.")

@st.cache_resource
def load_vectorstore():
    return build_vectorstore()


vectorstore = load_vectorstore()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Salut ! Pose-moi une question sur l’IA générative. "
                "Je répondrai à partir des pages Wikipédia indexées."
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_query = st.chat_input("Posez votre question…")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        status = st.status("🔎 Recherche des documents pertinents…", expanded=False)

        # 1) Retrieval
        with st.spinner("Recherche dans la base documentaire…"):
            docs = retrieve_documents(user_query, vectorstore, k=k)
            context = trim_context(docs, max_chars=max_chars)

        status.update(label="🧠 Génération de la réponse…", state="running", expanded=False)

        prompt_template = ChatPromptTemplate.from_template(
            """Tu es un assistant expert en intelligence artificielle.
Réponds en français.
Tu dois utiliser uniquement le contexte fourni.
Si le contexte ne suffit pas, dis-le clairement.
Fais une réponse claire et structurée.

Contexte :
{context}

Question :
{question}

Réponse :"""
        )

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            streaming=True,
        )

        chain = prompt_template | llm

        response_box = st.empty()
        full_response = ""

        with st.spinner("Le modèle rédige la réponse…"):
            for chunk in chain.stream(
                {
                    "context": context,
                    "question": user_query,
                }
            ):
                if chunk.content:
                    full_response += chunk.content
                    response_box.markdown(full_response)

        status.update(label="✅ Terminé", state="complete", expanded=False)

        with st.expander("Sources (Wikipédia)"):
            sources = sorted(
                {
                    d.metadata.get("source", "")
                    for d in docs
                    if d.metadata.get("source")
                }
            )
            st.markdown('<div class="sources">', unsafe_allow_html=True)
            for source in sources:
                st.write(source)
            st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )