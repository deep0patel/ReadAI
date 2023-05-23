import os.path
import os
import csv
import openai
import pandas as pd
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from werkzeug.utils import secure_filename, redirect
from wtforms import FileField, SubmitField
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma

from openai.embeddings_utils import cosine_similarity, get_embedding

openai_api_key = "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"

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

pdf_folder_path = 'ReadAI/static/File/pdf'


def pdf_upload(file, message):
    try:
        print("saving")
        file.save(os.path.join(pdf_folder_path,
                               secure_filename(file.filename)))
        print("saved")
        return True

    except (AttributeError, TypeError, ValueError, IOError) as e:
        message = 'Error uploading file: {}'.format(str(e)), 'error'
        return False


def pdf_process(message):
    # get a list of all files in the folder
    files = os.listdir(pdf_folder_path)
    # filter for PDF files
    pdf_files = [file for file in files if file.endswith('.pdf')]
    # check if any PDF files were found
    if len(pdf_files) == 0:
        message = message.append('No PDF files found in folder, upload again')
        return False
    else:
        # load the first PDF file found
        pdf_path = os.path.join(pdf_folder_path, pdf_files[0])
        loader = UnstructuredPDFLoader(pdf_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
        #
        # print(type(texts[0]))
        # print(texts[0])
        # print(texts[0].page_content)

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


def answer(query):
    search_term_vector = get_embedding(query, engine=embedding_model)

    csv_folder_path = 'ReadAI/static/File/csv'

    files = os.listdir(csv_folder_path)
    # filter for PDF files
    csv_files = [file for file in files if file.endswith('_emb.csv')]
    # check if any PDF files were found
    if len(csv_files) == 0:
        message = 'No emb_csv files found in folder, load again'
        return False
    else:

        # csv_path = os.path.join(csv_folder_path, csv_files[0])
        #
        # df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        # df["embedding"] = df["embedding"].apply(eval).apply(np.array)
        # # df['embedding'] = df['embedding'].apply(
        # #     lambda x: np.array(x.replace('[', '').replace(']', '').replace('\n', '').split(), dtype=float))
        # df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
        #
        # results = df.sort_values("similarities", ascending=False).head(5)
        #
        # llm = OpenAI(temperature=0, openai_api_key="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")
        #
        # chain = load_qa_chain(llm, chain_type="stuff")
        #
        # print('Reached till return')
        #
        # # return chain.run(input_documents=results, question=query)
        #
        # print(results)

        qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)

        result = qa({"query": query})

        return results
