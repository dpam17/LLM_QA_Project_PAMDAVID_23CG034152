import streamlit as st
import os
import requests
import re

# ----------------------------------------------------
# GLOBAL CONFIGURATION
# ----------------------------------------------------
# 1. Corrected HuggingFace API URL (Removed leading space and used the standard Inference API)
# This model is Mistral-7B-Instruct-v0.2
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# ----------------------------------------------------
# CORE FUNCTIONS
# ----------------------------------------------------

def preprocess_question(question):
    """
    Applies basic preprocessing: lowercase and punctuation removal, 
    as required by the project specifications.
    """
    # 1. Lowercase
    processed = question.lower()
    
    # 2. Punctuation removal (Removes non-word characters except whitespace)
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
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                # Extract and clean up the generated text
                answer = result[0].get('generated_text', 'No response generated').strip()
            else:
                answer = "Error: Invalid response structure from API."
            return answer
        
        # Handle API errors
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the API. Check your network connection."
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
    # Tries to get the key from environment variable/Streamlit secrets first
    api_key = st.text_input("Enter HuggingFace API Key", 
                            type="password", 
                            value=os.getenv("HUGGINGFACE_API_KEY", ""))
    st.markdown("---")
    st.markdown("### About")
    st.info("This Q&A system uses HuggingFace's Mistral-7B model to answer your questions. Make sure your API key is correct and active.")

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
        # Clear previous info and results
        processing_placeholder.empty()
        result_container.empty()
        
        with st.spinner("Processing your question and querying the LLM..."):
            
            # --- PART A Requirement: Preprocessing ---
            processed_question, tokens = preprocess_question(user_question)
            
            # Display preprocessing info
            with processing_placeholder.container():
                st.success("✅ Preprocessing Complete")
                st.write(f"**Original Question:** {user_question}")
                st.write(f"**Processed Question:** {processed_question}")
                st.write(f"**Tokens:** {', '.join(tokens)}")
                st.write(f"**Token Count:** {len(tokens)}")
            
            # --- PART A Requirement: Query LLM ---
            answer = query_llm(processed_question, api_key)
            
            # Display answer
            with result_container:
                st.subheader("💡 Answer (LLM API Response)")
                
                # Check for errors before displaying the final answer
                if answer.startswith("Error:") or answer.startswith("An unexpected error occurred:"):
                    st.error(answer)
                else:
                    st.success(answer)
                
                # Additional info (This is where the API_URL was fixed to be accessible)
                with st.expander("ℹ️ View Details"):
                    st.json({
                        "original_question": user_question,
                        "processed_question": processed_question,
                        "token_count": len(tokens),
                        "model": "mistralai/Mistral-7B-Instruct-v0.2",
                        "api_url_used": API_URL # FIX: API_URL is now global and accessible
                    })

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Built with Streamlit & HuggingFace API</div>", unsafe_allow_html=True)
