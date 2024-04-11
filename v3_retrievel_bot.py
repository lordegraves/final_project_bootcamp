# %%
import streamlit as st

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os

def get_vector_store():
    
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )
    
    return vector_store

def main():
    # Set the model name for our LLMs.
    OPENAI_MODEL = "gpt-3.5-turbo"
    # Store the API key in a variable.
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    
    st.title('Chat with Document')
    
    st.set_page_config(page_title="Ask Qdrant")
    st.header("Ask your remote database 💬")
    
    # create vector store
    vector_store = get_vector_store()
        
    #llm = OpenAI(temperature=0)
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)    
    retriever=vector_store.as_retriever()
    #chain = RetrievalQA.from_chain_type(llm, retriever=retriever )
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

