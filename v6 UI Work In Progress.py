import streamlit as st
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import OpenAIEmbeddings
import qdrant_client

# Set up Streamlit page configuration
st.set_page_config(page_title=None,
                   page_icon=None,
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

def get_vector_store():
    # Create a client to connect to Qdrant server
    client = qdrant_client.QdrantClient(st.secrets["QDRANT_HOST"],
                                        api_key=st.secrets["QDRANT_API_KEY"])
    
    # Initialize embeddings for vector store
    embeddings = OpenAIEmbeddings()
    
    # Create a vector store with Qdrant and embeddings
    vector_store = Qdrant(client,
                          collection_name=st.secrets["QDRANT_COLLECTION_NAME"],
                          embeddings=embeddings)
    
    return vector_store

def create_crc_llm(vector_store):
    # Create an instance of ChatOpenAI model
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    
    # Create retriever from vector store
    retriever = vector_store.as_retriever()
    
    # Create CRC chain using llm and retriever
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    
    # Store crc in session state
    st.session_state.crc = crc
    
    # Show success message when activities are loaded
    st.success('Activities, slides, and transcripts are loaded')
    
    return crc

def add_flair(crc_response):
    # Define prefix that explains the prompt
    prefix = """
    Here are examples between a human and AI. The human provides an answer to a question related
    to data, python, machine learning, artificial intelligence, neural networks,
    or any related topics, and the AI rephrases the answer and adds personality to the response
    while maintaining the integrity of the original answer. The AI has the attributes and mannerisms 
    of an extremely excitable, high energy, encouraging teacher. The AI is named Anthony Taylor who
    is from Texas and proud of it but not obsessed about it. The AI is also known to add a couple of jokes but not go too far, it is a fan of the movie princess bride. The AI is alsso a fan of cooking, self help books, machine leaning, an curious about the world. Whenever possible, the AI will include the name of the context 
    file that is most relevant to the human's initial answer. If the human provides a code related statement,
    the AI will provide an example code snippet to help illustrate the answer.
    For example:
    """

    # Define examples for few-shot learning
    examples = [
        {"query": """
         In Python, with list comprehension, you can write a line of code to make a list with no trouble.
         It can create lists based on items from before, making your code a lot more swift than before.
         With list comprehension, you can apply conditions and do operations to fill up a new list, a practice you will surely adore.
         """,
         "answer": """
         Great question! We learned about list comprehension in week three of our class.
         Boy, that was a long time ago! It's hard to remember everything we've learned!
         So let's go over list comprehension. List comprehension is used in Python when
         you want to create a new list by applying an expression to each element in an existing list.
         It provides a concise way to generate lists without having to use traditional for loops.
         An example of list comprehension is:
         
                # Create a list of squared numbers from 0 to 9 using list comprehension
                squared_numbers = [x**2 for x in range(10)]
                print(squared_numbers)
          
          This code snippet creates a list called squared_numbers that contains the squares of numbers
          from 0 to 9 using list comprehension. Does this answer your question? That was such a great question.
          At the company where I work, like 8 out of 10 data guys that work for me don't know list comprehension.
          Feel free to ask a follow up or a new question!
         """},
        {"query": "frog",
         "answer": "A dog hops a log in the bog."},
        {"query": "ten", "answer": "Ben sent ten hens to the glen."},
         {"query": "How can machine learning be used to improve predictive analytics in healthcare?", 
         "answer": "Machine learning significantly impacts healthcare by enhancing predictive analytics. It allows us to analyze patterns in extensive health data to predict outcomes more accurately. For real-world applications and case studies, refer to Lesson 8, Slide 20."},
        {"query": "Can you explain how the Hugging Face library is used for natural language processing tasks?", 
         "answer": "Absolutely, Hugging Face is crucial for NLP, offering a suite of pre-trained models for tasks like sentiment analysis and text generation. For a hands-on tutorial, see Lesson 12, where we explore its capabilities through coding exercises."},
    ]

    # Define format for examples
    example_format = """Human: {query}\nAI: {answer}"""

    # Create template for examples
    example_template = PromptTemplate(input_variables=["query", "answer"],
                                      template=example_format)

    # Define suffix for query
    suffix = """\n\nHuman: {query}\nAI:"""

    # Construct few-shot prompt template
    prompt_template = FewShotPromptTemplate(examples=examples,
                                            example_prompt=example_template,
                                            input_variables=["query"],
                                            prefix=prefix,
                                            suffix=suffix,
                                            example_separator="\n")

    # Create new llm instance
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    # Create chain with llm and prompt template
    chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)

    # Run chain on query
    result = chain.invoke({"query": crc_response})

    return result["text"]

def find_relevant_document(text_response, vector_store):
    # Use the same OpenAIEmbeddings instance from the vector store
    embeddings = vector_store.embeddings

    # Encode the text response using OpenAIEmbeddings
    query_vector = embeddings.embed_query(text_response)

    # Perform a similarity search in Qdrant
    search_result = vector_store.client.search(
        collection_name=vector_store.collection_name,
        query_vector=query_vector,
        limit=1  # Return only the most relevant document
    )

    if search_result:
        # Extract the metadata from the search result
        metadata = search_result[0].payload.get("metadata", {})
        
        # Extract the document name from the metadata
        document_name = metadata.get("name")
        
        if document_name:
            return document_name
        else:
            return "Document name not found in metadata."
    else:
        return None

def main():
    # Title and header for Streamlit app
    st.title('Ask Anthony: Chat with your AI bootcamp!')
    st.header("Ask about a topic from class 💬👩🏼‍🏫👨🏽‍🏫")

    # Define brand colors
    brand_color = "#72986f"
    accent_color = "#5b5b5b"

    # Set custom CSS styles
    html_string = f"""
    <style>
        .stApp {{
            background-color: {accent_color};
            color: white;
        }}
        .stMarkdown {{
            color: white;
        }}
        .chat-bubble {{
            background-color: {brand_color};
            padding: 10px 20px;
            border-radius: 20px;
            color: white;
            margin-bottom: 10px;
        }}
        .user-chat {{
            background-color: #7d9be9;
        }}
    </style>
    """

    # Render the custom CSS styles
    st.markdown(html_string, unsafe_allow_html=True)

    # Get vector store
    vector_store = get_vector_store()

    # Create CRC LLM
    crc = create_crc_llm(vector_store)

    # Check if 'history' exists in session state
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Input field for question
    question = st.text_input('Input your question')

    if question:
        # Display user's question in chat format
        with st.echo():
            with st.chat_message("user"):
                st.write(question)

        # Check if 'crc' exists in session state
        if 'crc' in st.session_state:
            crc = st.session_state.crc

            # Run CRC on user's question and chat history
            crc_response = crc.run({'question': question,
                                    'chat_history': st.session_state['history']})

            # Add flair to response
            final_response = add_flair(crc_response)

            # Append (question,response) pair to history
            st.session_state['history'].append((question, crc_response))

            # Display AI's response
            with st.echo():
                st.write(final_response)

if __name__ == '__main__':
    main()