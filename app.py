import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

st.title("AI Resume Screening System")

# Job description input
job_description = st.text_area("Enter Job Description")

# Upload resumes
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and job_description:

    resume_texts = []
    resume_names = []

    for file in uploaded_files:
        text = extract_text(file)
        resume_texts.append(text)
        resume_names.append(file.name)

    documents = [job_description] + resume_texts

    # ML Feature Extraction
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Similarity Calculation
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    results = list(zip(resume_names, similarity_scores[0]))
    results.sort(key=lambda x: x[1], reverse=True)

    st.subheader("Candidate Ranking")

    for name, score in results:
        st.write(f"{name} — Match Score: {score:.2f}")