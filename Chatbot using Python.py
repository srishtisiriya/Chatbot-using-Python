#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install torch')
get_ipython().system('pip install nltk')


# In[ ]:


import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the model to evaluation mode
model.eval()

# Function to generate the Catbot response
def generate_catbot_response(input_text):
    # Encode the input text and add batch dimension
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate a response (max length 50 tokens)
    response = model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the response
    output_text = tokenizer.decode(response[0], skip_special_tokens=True)
    
    # Add some "cat" style responses
    cat_responses = [
        "Meow meow! 🐾",
        "Purr... I'm a clever cat! 😸",
        "I'm a cat, not a robot, but I can still talk! 🐱",
        "Hiss! Just kidding... I'm here to help! 😺",
        "I can't help myself, I'm just too cute. 😽"
    ]
    
    # Randomly append a "cat" response to the output
    output_text += " " + random.choice(cat_responses)
    
    return output_text

# Chat loop to interact with the Catbot
def chat_with_catbot():
    print("Welcome to the AI-powered Catbot! Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Catbot: Bye bye! Purr... 😸")
            break
        
        response = generate_catbot_response(user_input)
        print(f"Catbot: {response}")

# Run the chat function
if __name__ == "__main__":
    chat_with_catbot()


# In[ ]:




