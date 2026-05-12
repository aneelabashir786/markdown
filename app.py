import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import fitz  # PyMuPDF for PDF processing
from docx import Document  # python-docx for Word files
import io
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face model path
MODEL_PATH = "aneelaBashir22f3414/document-to-markdown-generation"

@st.cache_resource
def load_model():
    """Load model from Hugging Face"""
    with st.spinner("Model load ho raha hai..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        model = model.to(device)
        model.eval()
        return tokenizer, model

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    doc = Document(io.BytesIO(file.read()))
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file):
    """Extract text from TXT file"""
    return file.read().decode("utf-8")

def convert_to_markdown(text, tokenizer, model):
    """Convert extracted text to markdown"""
    # Truncate long text (adjust max_length as needed)
    max_input_length = 512
    if len(text) > max_input_length * 4:
        text = text[:max_input_length * 4]
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            temperature=0.7,
            do_sample=True,
            num_beams=4
        )
    
    markdown_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return markdown_output

# Streamlit UI
st.set_page_config(
    page_title="Document to Markdown Converter",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document to Markdown Converter")
st.markdown("Upload PDF, DOCX, ya TXT file aur markdown mein convert karein!")

# Load model
tokenizer, model = load_model()

# File upload
uploaded_file = st.file_uploader(
    "Choose a file",
    type=['pdf', 'docx', 'txt'],
    help="PDF, DOCX, ya TXT file upload karein"
)

if uploaded_file is not None:
    # Show file info
    st.info(f"**File:** {uploaded_file.name}")
    st.info(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
    
    # Extract text based on file type
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner("Text extract ho raha hai..."):
        if file_type == 'pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == 'docx':
            text = extract_text_from_docx(uploaded_file)
        elif file_type == 'txt':
            text = extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported file type!")
            st.stop()
    
    # Show extracted text preview
    with st.expander("Extracted Text Preview"):
        st.text(text[:1000] + "..." if len(text) > 1000 else text)
    
    # Convert to markdown
    if st.button("Convert to Markdown", type="primary"):
        if text.strip():
            with st.spinner("Markdown mein convert ho raha hai..."):
                markdown_output = convert_to_markdown(text, tokenizer, model)
            
            # Display result
            st.success("Conversion complete!")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📝 Markdown Output")
                st.markdown(markdown_output)
            
            with col2:
                st.subheader("💾 Download")
                # Download button for markdown
                st.download_button(
                    label="Download Markdown",
                    data=markdown_output,
                    file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}.md",
                    mime="text/markdown"
                )
                
                # Copy to clipboard button (using JavaScript)
                st.markdown("""
                <button onclick="navigator.clipboard.writeText(document.getElementById('markdown-content').innerText)">
                    Copy to Clipboard
                </button>
                """, unsafe_allow_html=True)
        else:
            st.error("Koi text extract nahi hua! Please check your file.")

# Footer
st.markdown("---")
st.markdown(
    "**Model:** Document-to-Markdown | **Built with:** Hugging Face Transformers + Streamlit"
)
