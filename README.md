# 🤖 TP : Chatbot basé sur un RAG

Un chatbot basé sur un système **RAG (Retrieval-Augmented Generation)** qui répond aux questions des utilisateurs en s'appuyant sur un ensemble de pages Wikipédia liées à l’IA générative.

---

## 🚀 Fonctionnalités

* 🔎 Recherche d’informations depuis Wikipédia
* 🧠 Indexation avec embeddings (ChromaDB)
* 🤖 Génération de réponses avec un LLM
* 🧩 Architecture simple et pédagogique

---

## 🧠 Stack utilisée

* 🐍 Python
* 🔗 LangChain
* 🤖 OpenAI API
* 🧾 Wikipédia API
* 📦 ChromaDB

---

## ⚙️ Installation

### 1️⃣ Cloner le projet

```bash
git clone https://github.com/AyaDoukarr/TP_RAG_template.git
cd TP_RAG_template
```

### 2️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 🔑 Configuration

Créer le fichier `.env` à partir du template :

```bash
cp .env.template .env
```

Puis ajouter ta clé API :

```
OPENAI_API_KEY=ta_clef_api
```

---

## ▶️ Lancer le projet

```bash
python main.py
```

---

## 📁 Structure du projet

```
TP_RAG_template/
│── main.py
│── app.py
│── requirements.txt
│── README.md
│── .env.template
│── chroma_db/
```

---

## 🎯 Objectif pédagogique

Ce TP permet de comprendre :

* le fonctionnement du RAG
* l'utilisation des embeddings
* l’intégration d’un LLM dans une application Python

---

## 👤 Auteur

**Aya Doukarr**
