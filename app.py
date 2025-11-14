# CORRECTION KERAS - À EXÉCUTER EN PREMIER
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Imports standards
import re

# Pour importer les fichiers PDF
import requests
import PyPDF2

# Traitement du texte
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Modèle Reranker
from FlagEmbedding import FlagReranker

# Objet text retriever
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# LLM via HuggingFace Inference API (au lieu de llama-cpp)
from huggingface_hub import InferenceClient
import gradio as gr

"""# GESTION DE LA BASE DE DONNÉES

## Etape 1 : récupération des fichiers PDFs:
"""

# Chemin du dossier où l'on souhaite télécharger les fichiers
chemin_dossier = "./RAG_IPCC"
if not os.path.exists(chemin_dossier):
    os.makedirs(chemin_dossier)

# URLs des fichiers à télécharger
urls = { "6th_report": "https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_FullVolume.pdf" }

# Télécharger les fichiers dans le dossier (seulement s'ils n'existent pas déjà)
for name, url in urls.items():
    file_path = os.path.join(chemin_dossier, f"{name}.pdf")
    if not os.path.exists(file_path):
        print(f"Téléchargement de {name}...")
        response = requests.get(url)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"{name} a été téléchargé.")
    else:
        print(f"{name} existe déjà, téléchargement ignoré.")

"""## Etape 2 : Extraction du texte des fichiers PDF"""

# Chemin du dossier contenant les fichiers PDF
chemin_dossier = "./RAG_IPCC"

# Liste des fichiers PDF dans le dossier
fichiers_pdf = [f for f in os.listdir(chemin_dossier) if f.endswith('.pdf')]

# Liste pour stocker le texte extrait de chaque PDF
extracted_text = []

# Boucle à travers chaque fichier PDF
for pdf in fichiers_pdf:
    print(f"*** PROCESSING FILE : {pdf} ***")

    # Chemin complet du fichier PDF
    chemin_pdf = os.path.join(chemin_dossier, pdf)

    # Ouverture du fichier PDF en mode lecture binaire
    with open(chemin_pdf, 'rb') as file:
        # Création d'un objet de lecteur PDF
        pdf_reader = PyPDF2.PdfReader(file)

        # Boucle à travers chaque page du PDF
        for page_num in range(len(pdf_reader.pages)):
            # Extraction du texte de la page actuelle
            page = pdf_reader.pages[page_num]
            text = page.extract_text()

            # Ajout du texte extrait à la liste
            extracted_text.append({"document": pdf, "page": page_num, "content": text})

# Affichage du texte extrait
print(f"Extracted {len(extracted_text)} pages from PDFs")

"""## Etape 3 : Traitement du texte en chunks propres"""

#### FONCTIONS ####

# Segmentation du texte de base

def splitting_by_numer_of_words(text, chunk_size):
  """
  Découpe un texte en chunks de taille donnée (nombre de caractères).

  Args:
    text (str): Le texte à splitter.
    chunk_size (int): La taille souhaitée des chunks (nombre de mots).

  Returns:
    list: Une liste de chunks de texte.
  """
  chunks = []
  for phrase in text.split('\n'):
    words = phrase.split()
    for i in range(0, len(words), chunk_size):
      chunks.append(' '.join(words[i:i + chunk_size]))
  return chunks

# Fonction de splitting par phrase

def splitting_by_sentences(text):
  """
  Découpe un texte en chunks par phrases.

  Args:
    text (str): Le texte à découper.

  Returns:
    list: Une liste de chunks de texte (phrases).
  """
  sentences = []
  list_paragraph = text.split("\n")
  for paragraph in list_paragraph:
    list_sent = paragraph.split(".")
    sentences = sentences + list_sent
  return sentences

# Nettoyage du contenu de chaque chunk

special_chars = [" ", '-', '&', '(', ')', '_', ';', '†', '+', '–', "'", '!', '[', ']', "'", '́', '̀', '\u2009', '\u200b', '\u202f', '©', '£', '§', '°', '@', '€', '$', '\xa0', '~','\n','�']

def remove_char(text, char):
    """Remove each specific character from the text for each character in the chars list."""
    return text.replace(char, ' ')

def remove_chars(text, chars):
    """ Apply remove_char() function to text """
    for char in chars:
        text = remove_char(text, char)
    return text

def remove_multiple_white_spaces(text):
    """Remove multiple spaces."""
    text = re.sub(" +", " ", text)
    return text

def clean_text(text, special_chars=special_chars):
    """Generate a text without chars expect points and comma and multiple white spaces."""
    text = remove_chars(text, special_chars)
    text = remove_multiple_white_spaces(text)
    return text

# Filtrage des mots vides

def contains_mainly_digits(text, threshold=0.5):
    """
    Checks if a text string contains a high percentage of digits compared to letters.

    Args:
        text (str): The input text to analyze.
        threshold (float, optional): The threshold value for the proportion of digits to letters.
            Defaults to 0.5.

    Returns:
        bool: True if the proportion of digits in the text exceeds the threshold, False otherwise.
    """
    if not text:
        return False
    letters_count = 0
    nbs_count = 0
    for char in text:
        if char.isalpha():
            letters_count += 1
        elif char.isdigit():
            nbs_count += 1
    if letters_count + nbs_count > 0:
        digits_pct = (nbs_count / (letters_count + nbs_count))
    else:
        return True
    return digits_pct > threshold

def remove_mostly_digits_chunks(chunks, threshold=0.5):
  return [chunk for chunk in chunks if not contains_mainly_digits(chunk['content'])]

#### EXECUTION ####

# Split intelligent avec différents paramètres
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# Split pertinent qui garde la structure du document
chunks = []
for page_content in extracted_text:
  chunks_list = text_splitter.split_text(page_content['content'])

  for chunk in chunks_list:
    text=clean_text(chunk)
    chunks.append({"document": page_content['document'],
                   "page": page_content['page'],
                   "content": text})
chunks=remove_mostly_digits_chunks(chunks)
print(f"Created {len(chunks)} chunks after processing")

"""# IMPLEMENTATION DU MODELE DE RECHERCHE RETENU"""

class TextRetriever:
    def __init__(self, embedding_model_name="mixedbread-ai/mxbai-embed-large-v1", reranking_model_name="BAAI/bge-reranker-large"):
        """
        Initialise les modèles d'embedding et de reranking.

        Args:
            embedding_model_name (str): Nom du modèle d'embedding.
            reranking_model_name (str): Nom du modèle de reranking.
        """
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        print(f"Loading reranker model: {reranking_model_name}")
        self.reranker_model = FlagReranker(reranking_model_name, use_fp16=True)
        self.vector_database = None  # Initialisation de la base de données vectorielle à None

    def store_embeddings(self, chunks, path="./chroma_db"):
        """
        Stocke les embeddings des chunks de texte dans une base de données vectorielle.

        Args:
            chunks (list of str): Liste de chunks de texte à stocker.
            path (str): Chemin du répertoire où la base de données sera stockée.
        """
        print(f"Storing embeddings to {path}...")
        self.vector_database = Chroma.from_texts(chunks, embedding=self.embedding_model, persist_directory=path)
        print("Embeddings stored successfully")

    def load_embeddings(self, path):
        """
        Charge les embeddings depuis une base de données vectorielle.

        Args:
            path (str): Chemin du répertoire de la base de données.
        """
        print(f"Loading embeddings from {path}...")
        self.vector_database = Chroma(persist_directory=path, embedding=self.embedding_model)
        print("Embeddings loaded successfully")

    def get_best_chunks(self, query, top_k=3):
        """
        Recherche les meilleurs chunks correspondant à une requête.

        Args:
            query (str): Requête de recherche.
            top_k (int): Nombre de meilleurs chunks à retourner.

        Returns:
            list: Liste des meilleurs chunks correspondant à la requête.
        """
        best_chunks = self.vector_database.similarity_search(query, k=top_k)
        return best_chunks

    def rerank_chunks(self, query, chunks):
        """
        Retourne le chunk le plus pertinent pour une requête donnée.

        Args:
            query (str): Requête de recherche.
            chunks (list): Liste des chunks à re-classer.

        Returns:
            list: Liste des chunks triés par pertinence.
        """
        best_chunks = self.get_best_chunks(query, top_k=10)
        rerank_scores = []
        chunk_texts = [chunk.page_content if hasattr(chunk, 'page_content') else str(chunk) for chunk in best_chunks]
        for text in chunk_texts:
          score = self.reranker_model.compute_score([query, text])
          rerank_scores.append(score)

        return [x for _, x in sorted(zip(rerank_scores, best_chunks), reverse=True)]

    def get_context(self, query):
        """
        Retourne le chunk le plus pertinent pour une requête donnée.

        Args:
            query (str): Requête de recherche.

        Returns:
            str: Contenu du chunk le plus pertinent.
        """
        best_chunks = self.get_best_chunks(query, top_k=1)
        return best_chunks[0].page_content

print("Initializing TextRetriever...")
retriever=TextRetriever()

all_chunks=[]
for chunk in chunks:
  all_chunks.append(chunk['content'])

# Vérifier si la base de données existe déjà pour éviter de la recréer
db_path = "./chroma_db"
if os.path.exists(db_path):
    print("Loading existing embeddings database...")
    retriever.load_embeddings(db_path)
else:
    print("Creating new embeddings database...")
    retriever.store_embeddings(all_chunks, db_path)

"""# MODELE LLM

## Etape 1 : Generation d'une réponse avec HuggingFace Inference API
"""

# Initialiser le client d'inférence HuggingFace (modèle gratuit et léger)
# Utilisation de Mistral-7B-Instruct via l'API gratuite au lieu de le télécharger
print("Initializing HuggingFace Inference Client...")
llm_client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2")

## FONCTIONS

# Basic context function.
def get_context_from_query(query):
    chunks = retriever.get_best_chunks(query, 4)
    # Extraire le texte des chunks
    context_parts = []
    for chunk in chunks:
        if hasattr(chunk, 'page_content'):
            context_parts.append(chunk.page_content)
        else:
            context_parts.append(str(chunk))
    return "\n\n".join(context_parts)

"""## Etape 2 : Sauvegarde d'un historique limité de conversation"""

class ConversationHistoryLoader:

  def __init__(self, k):
    self.k=k
    self.conversation_history = []

  # Fonction qui permet créer un prompt (string) sur l'historique de conversation.
  def create_conversation_history_prompt(self):
    conversation = ''

    if self.conversation_history == None or len(self.conversation_history) == 0:
      return conversation
    else:
      for exchange in reversed(self.conversation_history):
        conversation = conversation + '\nHuman: '+exchange['Human']+'\nAI: '+exchange['AI']
      return conversation

  # Fonction qui permet de mettre à jour l'historique de conversation
  # à partir de la dernière query et la dernière réponse du LLM.
  def update_conversation_history(self, query, response):
    exchange = {'Human': query, 'AI': response}
    self.conversation_history.insert(0, exchange)

    if len(self.conversation_history) > self.k:
      self.conversation_history.pop()

# Fonction pour générer une réponse avec le contexte et l'historique
def generate_response_with_context(instruction, context, chat_history=""):
    """
    Génère une réponse en utilisant l'API HuggingFace Inference.
    """
    # Construire le prompt complet
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request using the context provided and the previous conversation.

Context: {context}

{chat_history}

Human: [INST] {instruction} [/INST]

AI:
"""
    
    try:
        # Appeler l API HuggingFace pour générer la réponse
        response = llm_client.text_generation(
            prompt,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2
        )
        
        # Nettoyer la réponse
        response = response.strip()
        response = re.sub(r"\[context\..*?\]", "", response)
        response = re.sub(r"Al:\s*", "", response)
        
        return response
    except Exception as e:
        print(f"Erreur lors de la génération: {str(e)}")
        return f"Désolé, une erreur s est produite: {str(e)}"

# Créer l instance de gestion d historique
ch = ConversationHistoryLoader(k=3)

# Fonction principale pour répondre aux questions
def get_response(query):
    try:
        # Obtenir le contexte pertinent
        context = get_context_from_query(query)
        
        # Générer la réponse avec contexte et historique
        chat_history = ch.create_conversation_history_prompt()
        response = generate_response_with_context(query, context, chat_history)
        
        # Mettre à jour l historique
        ch.update_conversation_history(query, response)
        
        return response
        
    except Exception as e:
        return f"Erreur: {str(e)}"

# Interface Gradio
print("Creating Gradio interface...")
iface = gr.Interface(
    fn=get_response, 
    inputs=gr.Textbox(lines=2, placeholder="Posez votre question sur le climat..."), 
    outputs=gr.Textbox(lines=5, label="Réponse"),
    title="🌍 RAG Chatbot - Questions Climatiques", 
    description="Posez vos questions sur le changement climatique basées sur les rapports IPCC.",
    examples=[
        "Quels sont les principaux impacts du réchauffement climatique ?",
        "Comment les océans sont-ils affectés par le changement climatique ?",
        "Quelles sont les solutions pour réduire les émissions ?"
    ]
)

# Lancer l application
if __name__ == "__main__":
    print("Launching Gradio app...")
    iface.launch(server_name="0.0.0.0", server_port=7860)
