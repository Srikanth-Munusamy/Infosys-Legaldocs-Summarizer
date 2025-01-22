import requests
import streamlit as st
from pdfminer.high_level import extract_text
import smtplib
from email.message import EmailMessage
from email_validator import validate_email, EmailNotValidError
import spacy
from collections import Counter
import heapq
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import matplotlib
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

matplotlib.rcParams['font.family'] = 'Arial'

# Load environment variables
load_dotenv()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize ChromaDB and SentenceTransformer model for embeddings
client = chromadb.Client()

try:
    # Attempt to get the collection
    collection = client.get_collection("legal_docs_collection")
except chromadb.errors.InvalidCollectionException:
    # If the collection does not exist, create it
    collection = client.create_collection("legal_docs_collection")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Email settings
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

# Google Sheets settings
SHEET_URL = os.getenv("SHEET_URL")
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client = gspread.authorize(credentials)
sheet = client.open_by_url(SHEET_URL).sheet1

# Functions

def extract_key_clauses(text):
    indicators = ["shall", "must", "obliged to", "required to", "responsible for", "liable for", "warrant", "guarantee"]
    return [sent.text.strip() for sent in nlp(text).sents if any(ind in sent.text.lower() for ind in indicators)]

def detect_hidden_risks(text):
    risks = [
        "dependency", "contingency", "subject to", "provided that",
        "unless", "in the event of", "without prejudice", "under no circumstances",
        "notwithstanding", "in the case of", "except as otherwise",
        "limited to", "may be", "shall not", "in accordance with", "at the discretion of"
    ]
    return [sent.text.strip() for sent in nlp(text).sents if any(risk in sent.text.lower() for risk in risks)]

def summarize_text(text, num_sentences=5):
    doc = nlp(text)
    word_freq = Counter(token.text.lower() for token in doc if token.is_alpha and not token.is_stop)
    sent_scores = {sent: sum(word_freq.get(word.text.lower(), 0) for word in sent) for sent in doc.sents}
    return " ".join(str(sent) for sent in heapq.nlargest(num_sentences, sent_scores, key=sent_scores.get))

def send_email(recipient_email, subject, body):
    try:
        msg = EmailMessage()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

def update_google_sheets(data):
    sheet.append_row(data)

def visualize_data(key_clauses, risks, summary):
    st.subheader("📊 Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.bar(["Key Clauses", "Risks"], [len(key_clauses), len(risks)], color=["skyblue", "salmon"])
        ax.set_title("Key Clauses vs. Risks")
        st.pyplot(fig)

    with col2:
        if summary:
            word_freq = Counter(summary.split())
            fig, ax = plt.subplots()
            sns.barplot(x=list(word_freq.keys()), y=list(word_freq.values()), ax=ax, palette="viridis")
            ax.set_title("Word Frequency")
            plt.xticks(rotation=45)
            st.pyplot(fig)

def setup_rag_model():
    embeddings = SentenceTransformerEmbeddings(embedding_model)
    vector_store = Chroma(client=client, collection_name="legal_docs_collection", embedding_function=embeddings)
    retriever = vector_store.as_retriever()
    return retriever

def get_answer_from_groq_api(question, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [ { "role": "user", "content": question } ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()
            answer = response_json['choices'][0]['message']['content']
            return answer.strip()
        else:
            st.error(f"Error with Groq API: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        st.error(f"Error with API request: {e}")
        return ""

def store_embeddings(text, doc_id):
    existing_docs = collection.get(ids=[doc_id])
    if existing_docs:
        return
    embeddings = embedding_model.encode([text])
    collection.add(
        documents=[text],
        embeddings=embeddings,
        metadatas=[{"doc_id": doc_id}],
        ids=[doc_id]
    )

def fetch_regulatory_updates():
    try:
        response = requests.get("https://run.mocky.io/v3/db70cbfe-6a57-42af-996c-2856de24a473")
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch regulatory updates.")
            return []
    except Exception as e:
        st.error(f"Error fetching regulatory updates: {e}")
        return []

def track_regulatory_updates(text, updates):
    affected_sections = []
    for update in updates.get("regulatory_updates", []):
        if isinstance(update, dict):
            section = update.get("section", "")
            sub_section = update.get("sub_section", "")
            update_text = update.get("update", "").lower()
            if section.lower() in text.lower() and sub_section.lower() in text.lower():
                affected_sections.append(f"Section: {section} Sub-Section: {sub_section} Update: {update_text}")
    return affected_sections

def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    st.title("📜 Legal Document Analyzer")

    st.sidebar.header("Options")
    uploaded_file = st.sidebar.file_uploader("Upload a legal document (PDF)", type="pdf")
    option = st.sidebar.radio(
        "Select analysis:",
        ["Summarize", "Extract Key Clauses", "Risk Detection", "Regulatory Updates", "Question Answering", "All"]
    )
    email = st.sidebar.text_input("Enter email to receive results:")
    send_email_button = st.sidebar.button("Send Results via Email")

    if uploaded_file:
        text = extract_text(uploaded_file)
        st.success("Document processed successfully!")

        store_embeddings(text, uploaded_file.name)

        summary, key_clauses, risks, affected_sections = "", [], [], []
        if option in ["Summarize", "All"]:
            summary = summarize_text(text)
            st.subheader("✍ Summary")
            st.write(summary)

        if option in ["Extract Key Clauses", "All"]:
            key_clauses = extract_key_clauses(text)
            st.subheader("📝 Key Clauses")
            st.write("\n".join(key_clauses))

        if option in ["Risk Detection", "All"]:
            risks = detect_hidden_risks(text)
            st.subheader("⚠️ Risks Detected")
            st.write("\n".join(risks))

        if option in ["Regulatory Updates", "All"]:
            updates = fetch_regulatory_updates()
            if updates:
                affected_sections = track_regulatory_updates(text, updates)
                st.subheader("📑 Regulatory Updates Tracker")
                st.write("\n".join(affected_sections))

        if option == "Question Answering":
            question = st.text_input("Ask a question about the document:")
            if question:
                api_key = os.getenv("GROQ_API_KEY")
                answer = get_answer_from_groq_api(question, api_key)
                st.subheader("📝 Answer")
                st.write(answer)

        visualize_data(key_clauses, risks, summary)
        update_google_sheets([uploaded_file.name, len(key_clauses), len(risks), summary[:100], len(affected_sections)])

        if send_email_button and email:
            try:
                validate_email(email)
                body = (
                    f"Summary:\n{summary}\n\nKey Clauses:\n{key_clauses}\n\nRisks:\n{risks}\n\n"
                    f"Affected Sections:\n{affected_sections}"
                )
                send_email(email, "Legal Document Analysis Results", body)
            except EmailNotValidError:
                st.error("Invalid email address.")

if __name__ == "__main__":
    main()
