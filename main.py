import os
import requests
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

PAGES = [
    "Intelligence_artificielle_générative",
    "Transformeur_génératif_préentraîné",
    "Google_Gemini",
    "Grand_modèle_de_langage",
    "ChatGPT",
    "LLaMA",
    "Réseaux_antagonistes_génératifs",
    "Apprentissage_auto-supervisé",
    "Apprentissage_par_renforcement",
    "DALL-E",
    "Midjourney",
    "Stable_Diffusion",
]


def get_wikipedia_page(title: str) -> str | None:
    url = "https://fr.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    headers = {"User-Agent": "RAG_project/0.0.1"}

    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract")


def build_vectorstore(persist_dir: str = "./chroma_db") -> Chroma:
    print("Téléchargement des pages Wikipedia...")
    docs: list[Document] = []

    for page in PAGES:
        content = get_wikipedia_page(page)
        if content and content.strip():
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": f"https://fr.wikipedia.org/wiki/{page}",
                        "title": page,
                    },
                )
            )

    print("Découpage en chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = splitter.split_documents(docs)

    print("Création des embeddings OpenAI + indexation dans Chroma...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def generate_rewrites(query: str) -> list[str]:
    """
    Génère 4 reformulations de la requête pour améliorer la recherche.
    """
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """Génère 4 reformulations différentes de la question utilisateur
pour améliorer la recherche sémantique dans une base documentaire Wikipédia.
Retourne uniquement les reformulations, une par ligne, sans numéros.

Question : {question}"""
    )

    msg = llm.invoke(prompt.format_messages(question=query))
    lines = [line.strip("-• \t") for line in msg.content.split("\n") if line.strip()]

    # sécurité
    cleaned = []
    seen = set()
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(line)

    if not cleaned:
        cleaned = [
            query,
            f"Définition de {query}",
            f"Explication de {query}",
            f"À quoi sert {query} ?",
        ]

    return cleaned[:4]


def retrieve_documents(query: str, vectorstore: Chroma, k: int = 5) -> list[Document]:
    """
    - reformulations
    - retrieval pour chaque reformulation
    - fusion + dédoublonnage
    - renvoie les k meilleurs chunks
    """
    rewrites = generate_rewrites(query)
    queries = [query] + rewrites

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    all_docs: list[Document] = []
    for q in queries:
        all_docs.extend(retriever.invoke(q))

    seen = set()
    unique_docs: list[Document] = []
    for d in all_docs:
        key = (d.metadata.get("source", ""), d.page_content[:200])
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    return unique_docs[:k]


def trim_context(docs: list[Document], max_chars: int = 4000) -> str:
    parts = []
    total = 0

    for d in docs:
        chunk = d.page_content.strip()
        if not chunk:
            continue

        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(chunk[:remaining])
            break

        parts.append(chunk)
        total += len(chunk)

    return "\n\n".join(parts)


def rag_answer(query: str, vectorstore: Chroma) -> None:
    print("Recherche des documents pertinents...")
    docs = retrieve_documents(query, vectorstore, k=5)

    context = trim_context(docs, max_chars=4000)

    prompt = ChatPromptTemplate.from_template(
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

    chain = prompt | llm

    print("Réponse (streaming) :\n")
    for chunk in chain.stream({"context": context, "question": query}):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print("\n\nSources utilisées :")
    sources = sorted({d.metadata.get("source", "") for d in docs if d.metadata.get("source")})
    for s in sources:
        print(s)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY manquante dans le fichier .env")

    print("Building vectorstore...")
    vectorstore = build_vectorstore()

    try:
        while True:
            print("-" * 50)
            print("Posez une question :")
            question = input("> ").strip()
            print()

            if question:
                rag_answer(question, vectorstore)

            print("\n")
    except KeyboardInterrupt:
        print("\nExiting...")