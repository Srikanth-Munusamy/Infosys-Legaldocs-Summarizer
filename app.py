import streamlit as st
from pdfminer.high_level import extract_text
import smtplib
from email.message import EmailMessage
from email_validator import validate_email, EmailNotValidError
import spacy
from collections import Counter
import heapq
#import gspread
#from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import seaborn as sns

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Email settings
SENDER_EMAIL = "srikanth28092004@gmail.com"
SENDER_PASSWORD = "woiwrszryzfheqze"

# Google Sheets settings
# SHEET_URL = "https://docs.google.com/spreadsheets/d/1p_SPUElNAfxUosCUWB5V2BLu9sfra6v7kWFNuxYSNXc/"
# scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# credentials = ServiceAccountCredentials.from_json_keyfile_name(r"", scope)

# client = gspread.authorize(credentials)
# sheet = client.open_by_url(SHEET_URL).sheet1

def extract_key_clauses(text):
    clause_indicators = ["shall", "must", "obliged to", "required to", "responsible for", "liable for", "warrant", "guarantee", "indemnify", "breach of"]
    doc = nlp(text)
    key_clauses = [sent.text.strip() for sent in doc.sents if any(indicator in sent.text.lower() for indicator in clause_indicators)]
    return key_clauses

def detect_hidden_risks(text):
    risk_terms = ["dependency", "condition precedent", "subsequent", "contingency", "subject to", "provided that"]
    doc = nlp(text.lower())
    risks_detected = [sent.text.strip() for sent in doc.sents if any(term in sent.text for term in risk_terms)]
    return risks_detected

def summarize_text(text, num_sentences=5):
    doc = nlp(text)
    sentences = list(doc.sents)
    word_frequencies = Counter([token.text.lower() for token in doc if token.is_alpha and not token.is_stop])
    sentence_scores = {sent: sum(word_frequencies.get(word.text.lower(), 0) for word in sent) for sent in sentences}
    summarized_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join([str(sentence) for sentence in summarized_sentences])

def send_email(recipient_email, subject, body):
    msg = EmailMessage()
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

# def update_google_sheets(data):
#     sheet.append_row(data)

def visualize_data(key_clauses, risks, summary):
    st.subheader("Visualizations")

    # Bar plot for key clauses vs risks
    if key_clauses or risks:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Key Clauses", "Risks"], [len(key_clauses), len(risks)], color=['skyblue', 'salmon'])
        ax.set_title("Key Clauses vs. Risks Detected")
        ax.set_ylabel("Count")
        plt.tight_layout()  # Adjust layout to prevent overlap
        st.pyplot(fig)  # Pass the figure object explicitly

    # Word frequency plot in the summary
    if summary:
        st.write("Word Frequency in Summary:")
        word_freq = Counter(summary.split())
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(word_freq.keys()), y=list(word_freq.values()), ax=ax, hue=list(word_freq.keys()), palette='viridis', legend=False)
        ax.set_title("Word Frequency in Summary")
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')  # Rotate and align x-tick labels
        plt.tight_layout()  # Adjust layout to prevent overlap
        st.pyplot(fig)  # Pass the figure object explicitly

def main():
    st.title("Legal Document Analyzer Dashboard")

    # Sidebar for file upload and option selection
    st.sidebar.header("Options")
    uploaded_file = st.sidebar.file_uploader("Upload a legal document (PDF)", type="pdf")
    option = st.sidebar.selectbox('Choose an option:', ('Summarize', 'Extract Key Clauses', 'Risk Detection', 'All'))

    summary, key_clauses, risks = "", [], []

    if uploaded_file is not None:
        try:
            text = extract_text(uploaded_file)
            st.sidebar.success("Document Text Extracted Successfully.")
        except Exception as e:
            st.sidebar.error(f"Error extracting text from PDF: {e}")
            return

        if option in ('Summarize', 'All'):
            summary = summarize_text(text)
            st.write("### Summary:")
            st.write(summary)

        if option in ('Extract Key Clauses', 'All'):
            key_clauses = extract_key_clauses(text)
            st.write("### Key Clauses:")
            st.write("\n".join(key_clauses))

        if option in ('Risk Detection', 'All'):
            risks = detect_hidden_risks(text)
            st.write("### Hidden Risks Detected:")
            st.write("\n".join(risks))

        # Visualize the data
        visualize_data(key_clauses, risks, summary)

        # Collect and display email functionality
        email = st.sidebar.text_input("Enter your email to receive the results:")
        if st.sidebar.button("Send Email"):
            try:
                v = validate_email(email)
                email = v["email"]
                subject = "Legal Document Analysis Results"
                body = f"""Summary:\n{summary}\n\nKey Clauses:\n{key_clauses}\n\nHidden Risks Detected:\n{risks}"""

                send_email(email, subject, body)
            except EmailNotValidError as e:
                st.sidebar.error(f"Invalid email: {e}")

        # Google Sheets Update
        # data = [uploaded_file.name, len(key_clauses), len(risks), summary[:100]]
        # update_google_sheets(data)

if __name__ == "__main__":
    main()
