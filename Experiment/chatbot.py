from IPython import get_ipython
from IPython.display import display
# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# %%

def load_model():
    """Load the Mistral-7B model and tokenizer."""
    token = "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"  
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", token=token)
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", token=token)
    return tokenizer, model

def generate_response(tokenizer, model, user_input, max_length=150):
    """Generate a response from the chatbot."""
    inputs = tokenizer.encode(user_input, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def provide_educational_insights():
    """Provide educational insights about sugar addiction."""
    return (
        "Did you know that excessive sugar intake can lead to health issues like diabetes, obesity, and heart disease? "
        "Reducing sugar gradually can improve energy levels and overall well-being."
    )

def suggest_healthy_alternatives():
    """Suggest healthy alternatives to manage sugar cravings."""
    return (
        "Try satisfying your sweet tooth with natural options like fruits, yogurt, or nuts. "
        "Staying hydrated and eating balanced meals can also help reduce cravings."
    )

def main():
    """Main function to run the chatbot."""
    print("Loading the Mistral-7B model. This might take a while...")
    load_model()

    print("Welcome to the Sugar Addiction Chatbot! How can I assist you today?")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["quit", "exit"]:
            print("Chatbot: Thank you for chatting. Take care of your health!")
            break
        elif "educate" in user_input.lower():
            print("Chatbot:", provide_educational_insights())
        elif "suggest" in user_input.lower():
            print("Chatbot:", suggest_healthy_alternatives())
        else:
            print("Chatbot:", generate_response(tokenizer, model, user_input))

if __name__ == "__main__":
    main()
