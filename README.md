# Chatbot RAG – IA générative

Ce projet implémente un chatbot basé sur un pipeline **RAG (Retrieval-Augmented Generation)**.  
L’objectif est de produire des réponses pertinentes à partir de sources externes (ici Wikipédia) plutôt que de dépendre uniquement du modèle de langage.

Le projet a été réalisé dans un cadre pédagogique, avec une attention particulière portée à la clarté de l’architecture et à la compréhension du fonctionnement d’un système RAG de bout en bout.


## ⚙️ Fonctionnement

Le pipeline suit les étapes suivantes :

- récupération de pages Wikipédia liées à l’IA générative  
- découpage des textes en chunks  
- génération d’embeddings  
- stockage dans une base vectorielle (ChromaDB)  
- recherche des passages les plus pertinents  
- génération d’une réponse contextualisée via un LLM  


## 🛠️ Stack technique

- **Python**
- **LangChain**
- **OpenAI API**
- **ChromaDB**
- **Wikipedia API**
- **Streamlit** (interface utilisateur)


## 🚀 Installation

Cloner le dépôt :

```bash
git clone https://github.com/AyaDoukarr/TP_RAG_template.git
cd TP_RAG_template
````

Installer les dépendances :

```bash
pip install -r requirements.txt
```


## 🔑 Configuration

Créer le fichier `.env` à partir du template :

```bash
cp .env.template .env
```

Ajouter ensuite ta clé API :

```
OPENAI_API_KEY=ta_clef_api
```

## ▶️ Lancer le projet

### 🔹 Mode script (console)

```bash
python main.py
```

### 🔹 Mode interface (Streamlit)

```bash
streamlit run app.py
```

Puis ouvrir dans le navigateur :

```
http://localhost:8501
```


## 📁 Structure du projet

```
TP_RAG_template/
│── main.py          # pipeline RAG (console)
│── app.py           # interface Streamlit
│── requirements.txt
│── .env.template
│── chroma_db/
```

---

## 📌 Notes

* Le projet est volontairement simple pour rester lisible
* Les réponses dépendent fortement des documents récupérés
* Certaines réponses peuvent être imprécises selon le contexte

---

## 👤 Auteur

**Aya Doukarr**





