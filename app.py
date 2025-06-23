import os
import requests
from flask import Flask, render_template, request, session

app = Flask(__name__)
# Set a secret key for session management. Replace this with a strong random key in production.
app.secret_key = os.urandom(24) # Or a fixed string like 'your_very_secret_key_here' for development

# --- Configuration ---
# Replace with your Hugging Face API Token (ensure it has 'read' access)
HF_API_TOKEN = "hf token"
# You can experiment with other versions like v0.2 or v0.3 if available and stable on the Inference API
MODEL_ID = "mistralai/Devstral-Small-2505_gguf"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# --- Helper Function to Query Hugging Face API ---
def query_hf_model(payload, retries=3, backoff_factor=0.5):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    for i in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60) # Increased timeout
            response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
            
            result = response.json()
            
            # The response structure can vary. For text generation, it's often a list.
            if isinstance(result, list) and result and "generated_text" in result[0]:
                return result[0]["generated_text"]
            # Sometimes it might be directly the generated text or other structures
            # Add more specific parsing if needed based on actual model output
            elif isinstance(result, dict) and "generated_text" in result: # Check for direct dict with key
                 return result["generated_text"]
            else:
                print(f"Unexpected API response structure: {result}")
                return "Error: Could not parse model response."

        except requests.exceptions.RequestException as e:
            print(f"API Request failed: {e}")
            if response is not None and response.status_code == 503 and "currently loading" in response.text.lower() and i < retries - 1:
                wait_time = (response.json().get("estimated_time", 20.0) * (backoff_factor ** i)) # Basic backoff for loading
                print(f"Model is loading. Retrying in {wait_time:.2f} seconds...")
                import time
                time.sleep(wait_time)
                continue # Retry
            elif response is not None:
                 print(f"Response status: {response.status_code}, Response text: {response.text}")
            if i == retries - 1: # Last attempt
                return f"Error: API request failed after {retries} retries. {e}"
        except Exception as e: # Catch other potential errors like JSONDecodeError
            print(f"An unexpected error occurred during API query: {e}")
            if i == retries - 1:
                return f"Error: An unexpected error occurred. {e}"
    return "Error: Model query failed after multiple retries."


def format_chat_prompt(history):
    """
    Formats the chat history into a prompt suitable for Mistral Instruct models.
    Example: <s>[INST] User 1 [/INST] Bot 1 </s>[INST] User 2 [/INST]
    """
    if not history:
        return ""

    prompt_parts = []
    # Start with <s> if it's the beginning of a new full conversation turn for the model
    # The API handles individual turns, so we build the context.
    # For the very first turn, we might not need <s> if the API prepends it or expects raw first INST.
    # However, to maintain context correctly across multiple turns, we build the full sequence.

    full_prompt = "<s>" # Often good to start the whole sequence this way
    for i, message in enumerate(history):
        if message['role'] == 'user':
            full_prompt += f"[INST] {message['content']} [/INST]"
        elif message['role'] == 'assistant':
            full_prompt += f" {message['content']}</s>" # Model response, then end of turn and start of new potential INST
            if i < len(history) -1 and history[i+1]['role'] == 'user': # If another user message follows
                 full_prompt += "<s>" # Start a new sequence for the next INST

    # If the last message was from the assistant, we don't need to prompt further.
    # If the last message was from the user, the prompt is ready for the model to generate the next part.
    # The API expects just the prompt that the *model* should complete.
    # So, if the last message is user, the prompt should end with [/INST]

    # Ensure the final prompt ends correctly for the model to respond to the last user message.
    if history[-1]['role'] == 'user' and not full_prompt.endswith("[/INST]"):
         # This case should ideally not happen if logic above is correct, but as a safeguard:
         # Find the last actual user message and ensure it's properly formatted.
         # For simplicity, the above loop should construct it correctly.
         pass # The loop should handle this.

    # The Inference API for text-generation typically expects `inputs` to be the prompt.
    # It doesn't explicitly manage "history" in separate fields for basic calls.
    return full_prompt


@app.route('/', methods=['GET', 'POST'])
def chat():
    # Initialize chat history in session if it doesn't exist
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        user_message = request.form.get('message')
        if user_message:
            session['chat_history'].append({'role': 'user', 'content': user_message})
            session.modified = True # Make sure the session is saved

            # Construct prompt from history
            # For Mistral, the prompt needs to be carefully formatted with [INST] and [/INST] tags
            # The Inference API for `text-generation` (which Mistral uses) expects a single string input.
            # We need to send the conversational history formatted into this single string.

            # Keep only the last N turns to avoid overly long prompts (adjust N as needed)
            MAX_HISTORY_TURNS_FOR_PROMPT = 5 # Each turn is a user msg + bot reply
            relevant_history = session['chat_history'][-(MAX_HISTORY_TURNS_FOR_PROMPT*2):]
            
            prompt_for_model = ""
            for turn in relevant_history:
                if turn['role'] == 'user':
                    prompt_for_model += f"[INST] {turn['content']} [/INST]"
                elif turn['role'] == 'assistant':
                    # Add the assistant's response, ensuring it's followed by the </s> token
                    # and the <s> token before the next [INST] if more turns follow.
                    prompt_for_model += f" {turn['content']}</s><s>" 
            
            # Remove trailing <s> if it exists, as the model needs to complete the last [/INST]
            if prompt_for_model.endswith("<s>"):
                prompt_for_model = prompt_for_model[:-3]


            if not prompt_for_model: # Should not happen if user_message exists
                bot_response_text = "Error: Could not generate prompt."
            else:
                payload = {
                    "inputs": prompt_for_model,
                    "parameters": { # Optional parameters
                        "max_new_tokens": 500, # Adjust as needed
                        "return_full_text": False, # We only want the newly generated part
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 50,
                        "repetition_penalty": 1.1, # Slight penalty for repetition
                    }
                }
                
                bot_response_raw = query_hf_model(payload)

                # The `query_hf_model` should return just the generated text part.
                # If `return_full_text` was true in payload, we'd need to strip the prompt.
                # Since it's false, the output *should* be just the new text.
                bot_response_text = bot_response_raw

                # Clean up response: Sometimes models might add their own EOS tokens or repeat parts of the prompt.
                # For Mistral, check if the response starts with the prompt itself (if return_full_text was mistakenly true or API behaves unexpectedly)
                # A more robust cleanup might be needed depending on observed model behavior.
                if bot_response_text.strip().startswith("[INST]") or bot_response_text.strip().startswith(user_message):
                    # This indicates the model might be echoing input or not respecting return_full_text=False
                    # Or, the response structure is different. For now, we'll take it as is.
                    # A common cleanup is to remove any partial [INST] tags if they appear.
                    pass


            session['chat_history'].append({'role': 'assistant', 'content': bot_response_text})
            session.modified = True

    return render_template('chat.html', chat_history=session['chat_history'])


if __name__ == '__main__':
    if HF_API_TOKEN == "YOUR_HUGGING_FACE_API_TOKEN":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: You need to set your Hugging Face API Token in app.py           !!!")
        print("!!! The chatbot will not work until you replace 'YOUR_HUGGING_FACE_API_TOKEN'!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    app.run(debug=True)