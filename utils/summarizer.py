from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os


def summarize_text(transcript):

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    texts = text_splitter.split_text(transcript)
    docs = [Document(page_content=t) for t in texts]

    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    # Create Prompt
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following content in a clear and concise way:\n\n{content}"
    )

    # Combine all chunks
    full_text = " ".join([doc.page_content for doc in docs])

    chain = prompt | llm

    response = chain.invoke({"content": full_text})

    return response.content, docs