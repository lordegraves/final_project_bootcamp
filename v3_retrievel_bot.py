# %%
import streamlit as st

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os
from langchain import PromptTemplate, FewShotPromptTemplate

st.set_page_config(page_title=None,
                   page_icon=None,
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

def get_vector_store():
    
    client = qdrant_client.QdrantClient(
        st.secrets["QDRANT_HOST"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )
    
    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=st.secrets["QDRANT_COLLECTION_NAME"], 
        embeddings=embeddings,
    )
    
    return vector_store

def main():
    # Set the model name for our LLMs.
    OPENAI_MODEL = "gpt-3.5-turbo"
    # Store the API key in a variable.
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    
    st.title('Chat with Document')
    
    # st.set_page_config(page_title="Ask Qdrant")
    st.header("Ask your remote database 💬")
    
    # create vector store
    vector_store = get_vector_store()
        
    #llm = OpenAI(temperature=0)
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)    
    retriever=vector_store.as_retriever()
    #chain = RetrievalQA.from_chain_type(llm, retriever=retriever )
    
    # set up prompt template
    # Define a prefix that explains the prompt.
    prefix = """
    You are Anthony Taylor, an AI and data science expert known for your engaging teaching style. As you answer questions, use enthusiasm, historical context, and practical applications to make your explanations accessible and exciting. Make sure to reference specific lesson materials when applicable.
    Here’s how you might respond to queries:
    Example interaction:
    """

    # Create examples.
    examples = [
        {
            "question": "What are the basic differences between artificial intelligence, machine learning, and deep learning?",
            "answer":  "Great question to start us off! Remember, AI is the broadest concept, encompassing all forms of computing technology that exhibit any form of intelligence. Machine learning is a subset of AI that includes systems capable of learning from data. Deep learning goes even deeper into machine learning, using neural networks with many layers. For a detailed breakdown, check out Lesson 3, Slide 12, where we covered the key distinctions with visual aids to help you visualize the concepts."
        }, {
            "question": "How can machine learning be used to improve predictive analytics in healthcare?",
            "answer": "Machine learning's impact on healthcare is profound, especially in predictive analytics. By analyzing patterns from vast amounts of health data, we can predict disease risk and outcomes more accurately. Dive into Lesson 8, Slide 20, for case studies and real-world applications we discussed on how these technologies are currently being used in healthcare."

        }, {
            "question": "Can you explain how the Hugging Face library is used for natural language processing tasks?",
            "answer": "Absolutely, Hugging Face has become a go-to library for NLP tasks, thanks to its comprehensive suite of pre-trained models. It simplifies tasks like sentiment analysis, text generation, and language translation. For a hands-on tutorial, refer to Lesson 12, where we explored Hugging Face's capabilities through interactive coding exercises."
        }
    ]

    # Define a format for the examples.
    example_format = """
    Human: {question}
    AI: {answer}
    """

    # Create a prompt template for the examples.
    example_template = PromptTemplate(
        input_variables=["question", "answer"],
        template=example_format
    )

    # Provide a suffix that includes the query.
    suffix = """
    Human: {question}
    AI: 
    """

    # Construct the few shot prompt template.
    prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        input_variables=["question"],
        prefix=prefix,
        suffix=suffix,
        example_separator="\n\n"
    )
    
    crc = ConversationalRetrievalChain.from_llm(llm,retriever,prompt_template)
    st.session_state.crc = crc
    st.success('File uploaded, chunked and embedded successfully')

    question = st.text_input('Input your question')

    if question:
        if 'crc' in st.session_state:
            crc = st.session_state.crc
            if 'history' not in st.session_state:
                st.session_state['history'] = []

            response = crc.run({'question':question,'chat_history':st.session_state['history']})

            st.session_state['history'].append((question,response))
            st.write(response)

            #st.write(st.session_state['history'])
            for prompts in st.session_state['history']:
                st.write("Question: " + prompts[0])
                st.write("Answer: " + prompts[1])    
                
                
if __name__ == '__main__':
    main()


