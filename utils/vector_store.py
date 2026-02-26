import os
from langchain_community.vectorstores import DeepLake
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


def store_documents(docs):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    db = DeepLake(
        dataset_path="./deeplake_dataset",
        embedding_function=embeddings,
        token=os.getenv("ACTIVELOOP_TOKEN"),
    )

    db.add_documents(docs)

    return db


def create_qa_chain(db):

    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}"
    )
    
    def format_docs(docs):
        return "\n".join([doc.page_content for doc in docs])
    
    qa_chain = (
        {"context": (lambda x: x["question"]) | retriever | format_docs, "question": lambda x: x["question"]}
        | prompt
        | llm
    )

    return qa_chain