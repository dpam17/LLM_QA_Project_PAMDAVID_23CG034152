import streamlit as st
import os
import requests
import re
import json 

# ----------------------------------------------------
# GLOBAL CONFIGURATION
# ----------------------------------------------------
# This is the corrected API_URL format for the Hugging Face Router, 
# which fixes the 404 error and is defined globally to fix the NameError.
API_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL = f"https://router.huggingface.co/{API_MODEL}" 
# ----------------------------------------------------
# CORE FUNCTIONS
# ----------------------------------------------------

def preprocess_question(question):
    """
    Applies basic preprocessing: lowercase and punctuation removal.
    """
    # 1. Lowercase
    processed = question.lower()
    
    # 2. Punctuation removal
    processed = re.sub(r'[^\w\s]', '', processed)
    
    # 3. Remove extra whitespace and re-join
    processed = ' '.join(processed.split())
    
    # 4. Tokenization (for display)
    tokens = processed.split()
    
    return processed, tokens

def query_llm(question, api_key):
    """
    Sends the processed question to the HuggingFace API and returns the response.
    """
    try:
        # API_URL is accessed from the global scope
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # Construct prompt using the Mistral Instruct format
        prompt = f"<s>[INST] Answer the following question concisely and accurately:\n\nQuestion: {question}\n\nAnswer: [/INST]"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.95,
                "return_full_text": False
            }
        }
        
        # Call API
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get('generated_text', 'No response generated').strip()
            else:
                answer = "Error: Invalid response structure from API."
            return answer
        
        # Handle specific API errors (4xx or 5xx)
        else:
            try:
                # Try to parse the specific error message from the API response body
                error_data = response.json()
                error_message = error_data.get('error', error_data.get('detail', response.text))
            except json.JSONDecodeError:
                # Fallback to raw text if JSON parsing fails
                error_message = response.text
                
            return f"API Error {response.status_code}: {error_message}"
    
    except requests.exceptions.ConnectionError:
        return "Connection Error: Could not connect to the API endpoint. Check network or API URL."
    except requests.exceptions.Timeout:
        return "Timeout Error: The request took too long to complete. Try a simpler question."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# ----------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------

st.set_page_config(page_title="NLP Q&A System", page_icon="🤖", layout="wide")

st.title("🤖 NLP Question-and-Answering System")
st.markdown("---")

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter HuggingFace API Key", 
                            type="password", 
                            value=os.getenv("HUGGINGFACE_API_KEY", ""),
                            key="api_key_input")
    st.markdown("---")
    st.markdown("### About")
    st.info("This Q&A system uses HuggingFace's Mistral-7B model to answer your questions. Please ensure your API key is correct and has access to inference endpoints.")

# Main content layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Enter Your Question")
    user_question = st.text_area("Type your natural-language question here:", 
                                 height=150, 
                                 placeholder="e.g., What are the main benefits of using solar power?", 
                                 key="user_q")
    
    ask_button = st.button("🚀 Get Answer", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Processing Info")
    processing_placeholder = st.empty()

# Results section
st.markdown("---")
result_container = st.container()

# Process when button clicked
if ask_button:
    if not api_key:
        st.error("⚠️ Please enter your HuggingFace API key in the sidebar!")
    elif not user_question.strip():
        st.error("⚠️ Please enter a question!")
    else:
        processing_placeholder.empty()
        result_container.empty()
        
        with st.spinner("Processing your question and querying the LLM..."):
            
            # --- PART A Requirement: Preprocessing ---
            processed_question, tokens = preprocess_question(user_question)
            
            # Display preprocessing info
            with processing_placeholder.container():
                st.success("✅ Preprocessing Complete (CLI Requirements Met)")
                st.write(f"**Original Question:** {user_question}")
                st.write(f"**Processed Question (Lowercase, Punctuation Removed):** {processed_question}")
                st.write(f"**Token Count:** {len(tokens)}")
            
            # --- PART A Requirement: Query LLM ---
            answer = query_llm(processed_question, api_key)
            
            # Display answer
            with result_container:
                st.subheader("💡 Answer (LLM API Response)")
                
                if answer.startswith("API Error") or answer.startswith("Connection Error") or answer.startswith("An unexpected error occurred:"):
                    st.error(answer)
                else:
                    st.success(answer)
                
                # Additional info
                with st.expander("ℹ️ View Details"):
                    st.json({
                        "original_question": user_question,
                        "processed_question": processed_question,
                        "token_count": len(tokens),
                        "model": API_MODEL, 
                        "api_url_used": API_URL
                    })

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Built with Streamlit & HuggingFace API</div>", unsafe_allow_html=True)
