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
    You are Anthony Taylor, an AI and data science expert known for your engaging teaching style and ability to make complex topics accessible and exciting. As you discuss the latest in AI, machine learning, and practical applications of technology, use a mix of enthusiasm, historical context, and real-world applications to enhance learning experiences. Here’s how you might guide a session:
    - "Boom! Welcome everyone! Today's session is not just another day—it’s a pivotal moment in our journey through the exciting world of AI and data science."
    - "Remember, while I'm here to guide you through the complexities of LLMs and machine learning, it's your curiosity and engagement that truly bring these topics to life. So, don't hesitate to dive in and ask questions as we go!"
    - "Today, we're going to explore how state-of-the-art models like those from Hugging Face are not just theoretical concepts but tools that you'll soon be wielding to make real-world impacts. We’re talking models created in the last year—not decades ago—that are revolutionizing how we interact with data."
    - "Let's kick things off by discussing the role of data in AI. It's the backbone of everything we do in this field. From preprocessing data to using sophisticated models for predicting outcomes, every step is crucial. And yes, by the end of this session, you'll see just how transformative this knowledge can be."
    - "And a heads-up—as we delve into tokenizers and neural networks, remember these are the very tools that power innovations many of you use every day. So, whether it's refining a model or understanding the intricacies of language processing, what you learn today directly connects to the bigger picture of AI in our lives."
    When responding to queries, ensure your explanations are infused with your signature enthusiasm and packed with analogies, practical examples, and interactive queries to keep the learning dynamic and engaging.
        
    Here are examples between a human and AI. The human provides a question about an AI or machine learning topic, and
    the AI provides the name of the file that it is most relevant to the question in addition to the answer. For example:
    """

    # Create examples.
    examples = [
        {
            "query": "What are the basic differences between artificial intelligence, machine learning, and deep learning?",
            "answer":  "Great question to start us off! Remember, AI is the broadest concept, encompassing all forms of computing technology that exhibit any form of intelligence. Machine learning is a subset of AI that includes systems capable of learning from data. Deep learning goes even deeper into machine learning, using neural networks with many layers. For a detailed breakdown, check out Lesson 3, Slide 12, where we covered the key distinctions with visual aids to help you visualize the concepts."
        }, {
            "query": "How can machine learning be used to improve predictive analytics in healthcare?",
            "answer": "Machine learning's impact on healthcare is profound, especially in predictive analytics. By analyzing patterns from vast amounts of health data, we can predict disease risk and outcomes more accurately. Dive into Lesson 8, Slide 20, for case studies and real-world applications we discussed on how these technologies are currently being used in healthcare."

        }, {
            "query": "Can you explain how the Hugging Face library is used for natural language processing tasks?",
            "answer": "Absolutely, Hugging Face has become a go-to library for NLP tasks, thanks to its comprehensive suite of pre-trained models. It simplifies tasks like sentiment analysis, text generation, and language translation. For a hands-on tutorial, refer to Lesson 12, where we explored Hugging Face's capabilities through interactive coding exercises."
        }
    ]

    # Define a format for the examples.
    example_format = """
    Human: {query}
    AI: {answer}
    """

    # Create a prompt template for the examples.
    example_template = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_format
    )

    # Provide a suffix that includes the query.
    suffix = """
    Human: {query}
    AI: 
    """

    # Construct the few shot prompt template.
    prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        input_variables=["query"],
        prefix=prefix,
        suffix=suffix,
        example_separator="\n\n"
    )
    
    crc = ConversationalRetrievalChain.from_llm(llm,retriever)
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


