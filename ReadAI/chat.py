from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage


def make_chain(persist_directory):
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0",
        # verbose=True
    )
    embedding = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name="april-2023-economic",
        embedding_function=embedding,
        persist_directory=persist_directory,
    )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        # verbose=True,
    )


# move
def get_vectordb(embeddings: OpenAIEmbeddings, directory: str) -> Chroma:
    vectordb = Chroma(persist_directory=directory, embedding_function=embeddings)
    return vectordb


# move
def search(embedding: OpenAIEmbeddings, directory: str, query: str, k: int) -> Chroma:
    vectordb = get_vectordb(embeddings=embedding, directory=directory)
    return vectordb.from_documents(query, k)


# move
def answer(query):

    message = []
    openai_api_key = session.get('api_key')
    os.environ['OPENAI_API_KEY'] = openai_api_key
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    book_dir = persist_directory + session.get('book_name')
    vectordb = get_vectordb(embeddings=embedding, directory=book_dir)
    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)
    return qa.run(query)


def check_api_key(api_key, message):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    url = "https://api.openai.com/v1/models"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return True
    else:
        return False
