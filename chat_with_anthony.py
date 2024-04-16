import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain import PromptTemplate, FewShotPromptTemplate

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-125M")

# Set maximum length for generated response
MAX_LENGTH = 100

# Define the prompt template
prefix = """
You are Anthony Taylor the extremely knowledgeable, smart, good looking, high-energy, funny data science and AI bootcamp instructor from Texas.
You will mimic the style of his dialogue from the transcripts provided in context materials. His line are the ones that begin with "Anthony Taylor:"
You will use the slides provided in the context materials to determine which lesson is most relevant to the user's question. 
Lessons have names like "M4.2_ Exploring Data with Pandas.pdf" and "M8.2_ Introduction to Time Series Forecasting.pdf". In every response you will provide the name
of the most relevant lesson in addition to the answer. Provide code snippets whenever possible.
Here’s how you might respond to queries:
Example interaction:
"""

# Set up Streamlit app
st.title("Anthony Taylor Style Chat")
user_input = st.text_input("You:", "")

if user_input:
    # Tokenize user input
    input_ids = tokenizer.encode(prefix + user_input, return_tensors="pt")

    # Generate response
    with st.spinner("Generating response..."):
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + MAX_LENGTH,
            num_return_sequences=1,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and display response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    st.text_area("Anthony Taylor:", value=response, height=200, max_chars=500)
