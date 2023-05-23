import os.path
import os

from flask import session
from langchain.embeddings.openai import OpenAIEmbeddings
from werkzeug.utils import secure_filename
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.vectorstores import Chroma
import requests



# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# folder paths
pdf_folder_path = 'ReadAI/static/File/pdf/'
persist_directory = 'ReadAI/static/File/vectors/'


# handles file upload
def pdf_upload(file, message):
    try:
        file_message = []
        pdf_list = get_pdf_list(file_message)
        if file.filename not in pdf_list or len(pdf_list) == 0:
            # saves file
            file.save(os.path.join(pdf_folder_path,
                                   secure_filename(file.filename)))
            process = pdf_process(file, message)

            if process:
                return True
        else:
            message.append("File with similar file name already exists ! Try renaming your file.")
            return False

    except (AttributeError, TypeError, ValueError, IOError) as e:
        message.append('Error uploading file: {}'.format(str(e)))
        message.append('error')

        return False

# to get all the existing pdfss
def get_pdf_list(message):
    # get a list of all files in the folder
    files = os.listdir(pdf_folder_path)
    # filter for PDF files
    pdf_files = [file for file in files if file.endswith('.pdf')]
    # check if any PDF files were found
    if len(pdf_files) == 0:
        message.append('No PDF files found in folder, upload again')
        return []
    else:
        message.append('Files loaded')
        return pdf_files


# to process the pdf
def pdf_process(file, message):

    openai_api_key = session.get('api_key')
    # functions
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

    try:
        loader = UnstructuredPDFLoader(pdf_folder_path + file.filename)
        message.append('File Loaded successfully')
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        message.append('File Split successfully')

        # saving each pdf in the form of vectordb
        vectordb = Chroma.from_documents(documents=texts, embedding=embedding,
                                         persist_directory=persist_directory + file.filename)
        vectordb.persist()
        vectordb = None
        message.append('File stored as vectors successfully')
        return True
    except() as e:
        message.append(e)
        return False


# loads the vector database as per book
def get_vectordb(embeddings: OpenAIEmbeddings, directory: str) -> Chroma:
    vectordb = Chroma(persist_directory=directory, embedding_function=embeddings)
    return vectordb



# answers the question.
def answer(query):

    message = []
    openai_api_key = session.get('api_key')
    os.environ['OPENAI_API_KEY'] = openai_api_key
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    book_dir = persist_directory + session.get('book_name')
    vectordb = get_vectordb(embeddings=embedding, directory=book_dir)
    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)
    return qa.run(query)


# to check the api key
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
