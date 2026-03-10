# TP – Chatbot RAG (IA générative)

Ce projet a été réalisé dans le cadre d’un TP autour des systèmes RAG (Retrieval-Augmented Generation).  
L’objectif est de construire un chatbot capable de répondre à des questions en s’appuyant sur des contenus externes plutôt que sur sa seule mémoire.

Ici, le chatbot utilise plusieurs pages Wikipédia liées à l’IA générative pour produire ses réponses.

---

## 🧩 Principe

Le fonctionnement est simple :

1. récupération de contenus depuis Wikipédia  
2. découpage des textes en morceaux  
3. transformation en embeddings  
4. stockage dans une base vectorielle (ChromaDB)  
5. récupération des passages pertinents  
6. génération de la réponse avec un modèle de langage  

Ce TP m’a permis de mieux comprendre concrètement le fonctionnement d’un pipeline RAG.



## 🛠️ Technologies utilisées

- Python  
- LangChain  
- OpenAI API  
- Wikipédia API  
- ChromaDB  



## ▶️ Installation

Cloner le repo :

```bash
git clone https://github.com/AyaDoukarr/TP_RAG_template.git
cd TP_RAG_template
````

Installer les dépendances :

```bash
pip install -r requirements.txt
```



## ⚙️ Configuration

Créer le fichier `.env` :

```bash
cp .env.template .env
```

Puis ajouter ta clé :

```
OPENAI_API_KEY=ta_clef
```



## ▶️ Lancer le projet

```bash
python main.py
```



## 📌 Remarques

* Le projet est volontairement simple pour rester pédagogique
* Les réponses dépendent du contenu récupéré depuis Wikipédia
* Certaines questions peuvent donner des réponses approximatives



## 👤 Auteur

**Aya Doukarr**

