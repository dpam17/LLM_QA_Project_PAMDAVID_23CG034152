import os
import requests

def preprocess_question(question):
    """
    Basic preprocessing: lowercase, remove extra spaces, basic punctuation handling
    """
    # Lowercase
    question = question.lower()
    
    # Remove extra whitespace
    question = ' '.join(question.split())
    
    # Basic tokenization info (just for display)
    tokens = question.split()
    
    print(f"\n[Preprocessing]")
    print(f"Original: {question}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}\n")
    
    return question

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

def main():
    """
    Main CLI application loop
    """
    print("=" * 60)
    print("NLP Question-and-Answering System (CLI)")
    print("=" * 60)
    
    # Get API key from environment or user input
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if not api_key:
        print("\nNo HUGGINGFACE_API_KEY found in environment.")
        api_key = input("Enter your HuggingFace API key: ").strip()
    
    if not api_key:
        print("Error: API key is required!")
        return
    
    print("\nType 'quit' or 'exit' to stop.\n")
    
    while True:
        # Get user question
        question = input("Enter your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not question:
            print("Please enter a valid question.\n")
            continue
        
        # Preprocess
        processed_question = preprocess_question(question)
        
        # Query LLM
        print("[Querying LLM...]")
        answer = query_llm(processed_question, api_key)
        
        # Display answer
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()