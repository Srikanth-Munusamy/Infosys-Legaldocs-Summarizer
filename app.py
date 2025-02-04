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
import re
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

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

# Enhanced Functions
def clean_text(text):
    """Clean text to remove numbers, section numbers (e.g., '2.3', '2.4'), and irrelevant fragments."""
    # Remove numerical patterns like '2.3', '2.4', '1.1' etc.
    text = re.sub(r'\d+(\.\d+)+', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove any non-alphabetical text, if needed
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text.strip()

#Summarization Function
def summarize_text(text, num_sentences=5):
    """Summarize text excluding numerical values and common stopwords."""
    doc = nlp(clean_text(text))  # Clean the text before summarization
    
    word_freq = Counter(
        token.text.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and not token.like_num
    )
    sent_scores = {
        sent: sum(word_freq.get(word.text.lower(), 0) for word in sent if not word.like_num)
        for sent in doc.sents
    }
    top_sentences = heapq.nlargest(num_sentences, sent_scores, key=sent_scores.get)
    
    # Return summary as a properly formatted paragraph
    return ". ".join(str(sent).strip() for sent in top_sentences) + "."


#key clauses Extraction Function
def extract_key_clauses(text):
    """Extract key clauses with improved detection logic."""
    indicators = ["shall", "must", "obliged to", "required to", "responsible for", "liable for", "warrant", "guarantee"]
    clauses = []
    for sent in nlp(text).sents:
        if any(ind in sent.text.lower() for ind in indicators):
            clauses.append(sent.text.strip())
    return sorted(set(clauses), key=clauses.index)

#Risk Detection Function
def detect_hidden_risks(text):
    """Detect hidden risks with more refined matching."""
    risks = [
        "dependency", "contingency", "subject to", "provided that", "unless",
        "in the event of", "without prejudice", "under no circumstances",
        "notwithstanding", "in the case of", "except as otherwise",
        "limited to", "may be", "shall not", "in accordance with", 
        "at the discretion of", "conditional upon", "exclusive of",
        "without limitation", "subject to change", "binding upon",
        "liable for", "force majeure", "to the extent permitted by law",
    ]
    risk_phrases = []
    for sent in nlp(text).sents:
        if any(risk in sent.text.lower() for risk in risks):
            risk_phrases.append(sent.text.strip())
    return sorted(set(risk_phrases), key=risk_phrases.index) 
    

#Regulatory updates Fetch
def fetch_regulatory_updates():
    """Fetch regulatory updates with robust error handling."""
    try:
        response = requests.get("https://run.mocky.io/v3/db70cbfe-6a57-42af-996c-2856de24a473")
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to fetch regulatory updates: Check the API response.")
            return []
    except Exception as e:
        st.error(f"Error fetching regulatory updates: {e}")
        return []

#Track regulatory updates Function
def track_regulatory_updates(text, updates):
    """Track regulatory updates and identify affected sections."""
    affected_sections = []
    for update in updates.get("regulatory_updates", []):
        if isinstance(update, dict):
            section = update.get("section", "")
            sub_section = update.get("sub_section", "")
            update_text = update.get("update", "").lower()
            if section.lower() in text.lower() and sub_section.lower() in text.lower():
                affected_sections.append(f"Section: {section} Sub-Section: {sub_section} Update: {update_text}")
    return affected_sections

#Qustion & answering Funciton
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

#Data Visualization Function
def visualize_data(key_clauses, risks, summary):
    """Enhanced visualization with dynamic scaling and improved clarity."""
    st.subheader("📊 Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.bar(["Key Clauses", "Risks"], [len(key_clauses), len(risks)], color=["skyblue", "salmon"])
        ax.set_title("Key Clauses vs. Risks")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        if summary:
            word_freq = Counter(word for word in summary.split() if len(word) > 3)
            fig, ax = plt.subplots()
            sns.barplot(x=list(word_freq.keys()), y=list(word_freq.values()), ax=ax, palette="viridis")
            ax.set_title("Word Frequency")
            plt.xticks(rotation=45)
            st.pyplot(fig)

#Create pdf Function
def create_pdf(summary=None, key_clauses=None, risks=None, affected_sections=None, filename="analysis_report.pdf"):
    """Create a PDF with the provided content."""
    # Create PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font('DejaVu', '', './DejaVuSans.ttf', uni=True)
    pdf.set_font("DejaVu", size=14) 

    # Title
    pdf.cell(200, 10, txt="Legal Document Analysis Report", ln=True, align="C")
    pdf.ln(10)

    # Summary
    if summary:
        pdf.set_font("DejaVu", size=14)  
        pdf.cell(200, 10, txt="Summary:", ln=True)
        pdf.set_font("DejaVu", size=12)  
        pdf.multi_cell(0, 10, txt=summary + "\n\n")

    # Key Clauses
    if key_clauses:
        pdf.set_font("DejaVu", size=14) 
        pdf.cell(200, 10, txt="Key Clauses:", ln=True)
        pdf.set_font("DejaVu", size=12)  
        for clause in key_clauses:
            pdf.multi_cell(0, 10, txt=f"- {clause}\n")
        pdf.ln(5)

    # Risks Detected
    if risks:
        pdf.set_font("DejaVu", size=14) 
        pdf.cell(200, 10, txt="Risks Detected:", ln=True)
        pdf.set_font("DejaVu", size=12)  
        for risk in risks:
            pdf.multi_cell(0, 10, txt=f"- {risk}\n")
        pdf.ln(5)

    # Regulatory Updates
    if affected_sections:
        pdf.set_font("DejaVu", size=14)  
        pdf.cell(200, 10, txt="Regulatory Updates:", ln=True)
        pdf.set_font("DejaVu", size=12)  
        for section in affected_sections:
            pdf.multi_cell(0, 10, txt=f"- {section}\n")
        pdf.ln(5)

    # Visualizations
    if key_clauses or risks or summary:
        visualization_files = []

        if key_clauses and risks:
            fig, ax = plt.subplots()
            ax.bar(["Key Clauses", "Risks"], [len(key_clauses), len(risks)], color=["skyblue", "salmon"])
            ax.set_title("Key Clauses vs. Risks")
            ax.set_ylabel("Count")
            visualization_file = "key_clauses_vs_risks.png"
            fig.savefig(visualization_file, format="png")
            visualization_files.append(visualization_file)
            plt.close(fig)

        # Save Word Frequency bar plot for the summary
        if summary:
            word_freq = Counter(word for word in summary.split() if len(word) > 3)
            fig, ax = plt.subplots()
            sns.barplot(x=list(word_freq.keys()), y=list(word_freq.values()), ax=ax, palette="viridis")
            ax.set_title("Word Frequency")
            plt.xticks(rotation=45)
            visualization_file = "word_frequency.png"
            fig.savefig(visualization_file, format="png")
            visualization_files.append(visualization_file)
            plt.close(fig)

        # Add the saved images to the PDF
        for img_file in visualization_files:
            pdf.add_page()
            pdf.set_font("DejaVu", size=14)
            pdf.cell(200, 10, txt=f"Visualization: {os.path.basename(img_file)}", ln=True, align="C")
            pdf.image(img_file, x=10, y=30, w=180) 
            pdf.ln(5)
            os.remove(img_file) 

    
    pdf.output(filename)

    return filename

#Update google sheets Function
def update_google_sheets(data):
    """Update Google Sheets with error handling."""
    try:
        sheet.append_row(data)
    except Exception as e:
        st.error(f"Error updating Google Sheets: {e}")


#Send Email Function
def send_email(recipient_email, subject, body,attachment_filename=None):
    """Send email with robust error handling."""
    try:
        msg = EmailMessage()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.set_content(body)

        if attachment_filename:
            with open(attachment_filename, "rb") as f:
                file_data = f.read()
                file_name = os.path.basename(attachment_filename)
                msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        st.success("Email sent successfully!")
    except smtplib.SMTPAuthenticationError:
        st.error("Authentication error: Please check your email credentials.")
    except Exception as e:
        st.error(f"Error sending email: {e}")


#main Function
def main():
    """Main function for Streamlit app."""
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    st.title("📜 Legal Document Analyzer")

    st.sidebar.header("Options")
    uploaded_file = st.sidebar.file_uploader("Upload a legal document (PDF)", type="pdf")
    option = st.sidebar.radio(
        "Select analysis:",
        ["Summarize", "Extract Key Clauses", "Risk Detection", "Regulatory Updates", "Question Answering", "Visualizations","All"]
    )
    email = st.sidebar.text_input("Enter email to receive results:")
    send_email_button = st.sidebar.button("Send Results via Email")

    # Initialize variables
    summary = ""
    key_clauses = []  
    risks = []
    affected_sections = []

    if uploaded_file:
        text = extract_text(uploaded_file)
        st.success("Document processed successfully!")

        summary = summarize_text(text)
        key_clauses = extract_key_clauses(text)  
        risks = detect_hidden_risks(text)

        if option in ["Summarize", "All"]:
            #summary = summarize_text(text)
            st.subheader("✍ Summary")
            st.write(summary)

        if option in ["Extract Key Clauses", "All"]:
           
            st.subheader("📝 Key Clauses")
            st.write("\n".join(key_clauses))

        if option in ["Risk Detection", "All"]:
            #risks = detect_hidden_risks(text)
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

        if option in ["Visualizations","All"]:
            # Visualize data
            visualize_data(key_clauses, risks, summary)

        # Update Google Sheets with document stats
        update_google_sheets([uploaded_file.name, len(key_clauses), len(risks), summary[:100], len(affected_sections)])

        # Send email with results if requested
        if send_email_button and email:
            pdf_filename = create_pdf(summary, key_clauses, risks, affected_sections)
            try:
                validate_email(email)
                body = (
                    "The detailed analysis of the uploaded document is attached with this mail."
                 )
                send_email(email, "Legal Document Analysis Results", body, pdf_filename)
            except EmailNotValidError:
                st.error("Invalid email address.")
            finally:
                os.remove('./analysis_report.pdf')

if __name__ == "__main__":
    main()
