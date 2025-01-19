import streamlit as st
from pdfminer.high_level import extract_text
import smtplib  # Add this import at the top of your code
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
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

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
    risks = ["dependency", "contingency", "subject to", "provided that"]
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

def setup_qa_model():
    # Load the question-answering model and tokenizer
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return qa_pipeline

def get_answer_from_qa(question, text, qa_pipeline):
    # Get the answer to the question from the document using the QA model
    answer = qa_pipeline(question=question, context=text)
    return answer['answer']


# Main app
def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    st.title("📜 Legal Document Analyzer")

    st.sidebar.header("Options")
    uploaded_file = st.sidebar.file_uploader("Upload a legal document (PDF)", type="pdf")
    option = st.sidebar.radio("Select analysis:", ["Summarize", "Extract Key Clauses", "Risk Detection", "Question Answering","All"])
    email = st.sidebar.text_input("Enter email to receive results:")
    send_email_button = st.sidebar.button("Send Results via Email")

    if uploaded_file:
        text = extract_text(uploaded_file)
        st.success("Document processed successfully!")

        summary, key_clauses, risks = "", [], []
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

        # Setup the QA model
        if option == "Question Answering":
            qa_pipeline = setup_qa_model()
            question = st.text_input("Ask a question about the document:")
            if question:
                answer = get_answer_from_qa(question, text, qa_pipeline)
                st.subheader("📝 Answer")
                st.write(answer)

        # Visualize results
        visualize_data(key_clauses, risks, summary)

        # Update Google Sheets
        update_google_sheets([uploaded_file.name, len(key_clauses), len(risks), summary[:100]])

        # Send email
        if send_email_button and email:
            try:
                validate_email(email)
                body = f"Summary:\n{summary}\n\nKey Clauses:\n{key_clauses}\n\nRisks:\n{risks}"
                send_email(email, "Legal Document Analysis Results", body)
            except EmailNotValidError:
                st.error("Invalid email address.")

if __name__ == "__main__":
    main()
