import streamlit as st
import os
import requests

def preprocess_question(question):
    """
    Basic preprocessing: lowercase, tokenization
    """
    # Lowercase
    processed = question.lower()
    
    # Remove extra whitespace
    processed = ' '.join(processed.split())
    
    # Tokenization
    tokens = processed.split()
    
    return processed, tokens

def query_llm(question, api_key):
    """
    Send question to HuggingFace API and get response
    """
    try:
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # Construct prompt
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
                answer = result[0].get('generated_text', 'No response generated')
            else:
                answer = str(result)
            return answer
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="NLP Q&A System", page_icon="🤖", layout="wide")

st.title("🤖 NLP Question-and-Answering System")
st.markdown("---")

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter HuggingFace API Key", type="password", value=os.getenv("HUGGINGFACE_API_KEY", ""))
    st.markdown("---")
    st.markdown("### About")
    st.info("This Q&A system uses HuggingFace's Mistral-7B model to answer your questions. Get your free API key at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Enter Your Question")
    user_question = st.text_area("Type your question here:", height=150, placeholder="e.g., What is machine learning?")
    
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
        with st.spinner("Processing your question..."):
            # Preprocess
            processed_question, tokens = preprocess_question(user_question)
            
            # Display preprocessing info
            with processing_placeholder.container():
                st.success("✅ Preprocessing Complete")
                st.write(f"**Original Question:** {user_question}")
                st.write(f"**Processed Question:** {processed_question}")
                st.write(f"**Tokens:** {', '.join(tokens)}")
                st.write(f"**Token Count:** {len(tokens)}")
            
            # Query LLM
            answer = query_llm(processed_question, api_key)
            
            # Display answer
            with result_container:
                st.subheader("💡 Answer")
                st.success(answer)
                
                # Additional info
                with st.expander("ℹ️ View Details"):
                    st.json({
                        "original_question": user_question,
                        "processed_question": processed_question,
                        "token_count": len(tokens),
                        "model": "mistralai/Mistral-7B-Instruct-v0.2"
                    })

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Built with Streamlit & HuggingFace API</div>", unsafe_allow_html=True)