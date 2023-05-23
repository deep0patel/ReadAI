import os.path
import os
import csv

import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from werkzeug.utils import secure_filename
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.vectorstores import Chroma
from openai.embeddings_utils import cosine_similarity, get_embedding

os.environ['OPENAI_API_KEY'] = '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'
openai_api_key = "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"

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# folder paths
pdf_folder_path = 'ReadAI/static/File/pdf'
persist_directory = 'ReadAI/static/File/vectors'

# functions
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)


def pdf_upload(file, message):
    try:
        # saves file
        file.save(os.path.join(pdf_folder_path,
                               secure_filename(file.filename)))
        return True

    except (AttributeError, TypeError, ValueError, IOError) as e:
        message = 'Error uploading file: {}'.format(str(e)), 'error'
        return False


def get_pdf_list(message):
    # get a list of all files in the folder
    files = os.listdir(pdf_folder_path)
    # filter for PDF files
    pdf_files = [file for file in files if file.endswith('.pdf')]
    # check if any PDF files were found
    if len(pdf_files) == 0:
        message = message.append('No PDF files found in folder, upload again')
        return False


def pdf_process(file, message):
    # # get a list of all files in the folder
    # files = os.listdir(pdf_folder_path)
    # # filter for PDF files
    # pdf_files = [file for file in files if file.endswith('.pdf')]
    # # check if any PDF files were found
    # if len(pdf_files) == 0:
    #     message = message.append('No PDF files found in folder, upload again')
    #     return False
    #
    # else:
    #     # load the first PDF file found
    #     pdf_path = os.path.join(pdf_folder_path, pdf_files[0])
    loader = UnstructuredPDFLoader(file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None

    csv_file = 'ReadAI/static/File/csv/' + pdf_files[0].replace(".pdf", "") + '.csv'

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['', 'page_content'])

        i = 0
        for row in texts:
            writer.writerow([i, row.page_content])
            i += 1
    return True


def csv_to_emb(message):
    csv_folder_path = 'ReadAI/static/File/csv'
    files = os.listdir(csv_folder_path)
    # filter for PDF files
    csv_files = [file for file in files if file.endswith('.csv')]
    # check if any PDF files were found
    if len(csv_files) == 0:
        message.append('No csv found in folder.')
        return False
    else:

        csv_file = csv_file = 'ReadAI/static/File/csv/' + csv_files[0]
        df = pd.read_csv(csv_file, index_col=0)
        df = df[["page_content"]]
        # df["text"] = (df.page_content.str.strip().replace("/n", " "))
        df['embedding'] = df.page_content.apply(lambda x: get_embedding(x, engine=embedding_model))
        df.to_csv(csv_file.replace(".csv", "") + '_emb.csv')
        return True


def get_vectordb(embeddings: OpenAIEmbeddings, directory: str) -> Chroma:
    vectordb = Chroma(persist_directory=directory, embedding_function=embeddings)
    return vectordb


def search(embeddings: OpenAIEmbeddings, directory: str, query: str, k: int) -> Chroma:
    vectordb = get_vectordb(embeddings=embeddings, directory=directory)
    return vectordb.from_documents(query, k)


def answer(query):
    vectordb = get_vectordb(embeddings=embedding, directory=persist_directory)
    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)

    return qa.run(query)
