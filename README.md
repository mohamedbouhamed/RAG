---
title: "RAG Chatbot - Questions Climatiques"
emoji: "🌍"
colorFrom: "blue"
colorTo: "green"
sdk: "gradio"
sdk_version: "5.38.2"
app_file: "app.py"
pinned: false
---

# RAG Chatbot - Questions Climatiques

Application de chatbot basée sur RAG (Retrieval-Augmented Generation) pour répondre aux questions sur le changement climatique en utilisant les rapports IPCC.

## Fonctionnalités

- **Extraction et traitement automatique** des rapports IPCC en PDF
- **Recherche sémantique** avec embeddings de haute qualité (mixedbread-ai/mxbai-embed-large-v1)
- **Reranking** pour améliorer la pertinence des résultats (BAAI/bge-reranker-large)
- **Génération de réponses** via l'API HuggingFace Inference (Mistral-7B-Instruct)
- **Historique de conversation** pour des échanges contextualisés
- **Interface Gradio** intuitive

## Architecture

1. **Téléchargement des PDFs**: Récupération automatique des rapports IPCC
2. **Extraction du texte**: Parsing des PDFs avec PyPDF2
3. **Chunking intelligent**: Découpage en morceaux de 500 caractères avec overlap
4. **Embeddings**: Vectorisation avec sentence-transformers
5. **Base de données vectorielle**: Stockage avec ChromaDB
6. **Retrieval**: Recherche des chunks pertinents + reranking
7. **Génération**: Réponse contextualisée via Mistral-7B-Instruct

## Optimisations pour HF Spaces

- **Pas de téléchargement de modèle LLM**: Utilisation de l'API Inference HuggingFace (gratuite)
- **Cache intelligent**: Les PDFs et embeddings sont sauvegardés pour éviter de les recréer
- **Modèles légers**: Pas besoin de GPU
- **CPU uniquement**: Fonctionne sur le tier gratuit de HF Spaces

## Utilisation locale

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python app.py
```

L'application sera accessible sur `http://localhost:7860`

## Exemples de questions

- "Quels sont les principaux impacts du réchauffement climatique ?"
- "Comment les océans sont-ils affectés par le changement climatique ?"
- "Quelles sont les solutions pour réduire les émissions ?"

## Technologies utilisées

- **Gradio**: Interface utilisateur
- **LangChain**: Pipeline RAG
- **Sentence Transformers**: Embeddings sémantiques
- **ChromaDB**: Base de données vectorielle
- **FlagEmbedding**: Reranking des résultats
- **HuggingFace Inference API**: Génération de texte avec Mistral-7B
