import os.path
import os
import re
import pdfplumber
import PyPDF4
import sys
from flask import session
from typing import Callable, List, Tuple, Dict
from langchain.embeddings.openai import OpenAIEmbeddings
from werkzeug.utils import secure_filename
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.docstore.document import Document
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


def extract_metadata_from_pdf(file_path: str) -> dict:
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF4.PdfFileReader(pdf_file)  # Change this line
        metadata = reader.getDocumentInfo()
        return {
            "title": metadata.get("/Title", "").strip(),
            "author": metadata.get("/Author", "").strip(),
            "creation_date": metadata.get("/CreationDate", "").strip(),
        }


def extract_pages_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    """
    Extracts the text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A list of tuples containing the page number and the extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with pdfplumber.open(file_path) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text.strip():  # Check if extracted text is not empty
                pages.append((page_num + 1, text))
    return pages


def parse_pdf(file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    """
    Extracts the title and text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A tuple containing the title and a list of tuples with page numbers and extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata = extract_metadata_from_pdf(file_path)
    pages = extract_pages_from_pdf(file_path)

    return pages, metadata


def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(
        pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]
) -> List[Tuple[int, str]]:
    cleaned_pages = []
    for page_num, text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((page_num, text))
    return cleaned_pages


def text_to_docs(text: List[str], metadata: Dict[str, str]) -> List[Document]:
    """Converts list of strings to a list of Documents with metadata."""
    doc_chunks = []

    for page_num, page in text:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": f"p{page_num}-{i}",
                    **metadata,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks


# Returns boolean upon successful pdf upload
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


# Returns pdf list
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


def pdf_process(file, message):
    openai_api_key = session.get('api_key')
    # functions
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

    try:

        # Step 1 to generate path
        file_path = pdf_folder_path + file.filename
        # Step 2: Parse PDF
        raw_pages, metadata = parse_pdf(file_path)
        # Step 3 creating clean text chunks
        cleaning_functions = [
            merge_hyphenated_words,
            fix_newlines,
            remove_multiple_newlines,
        ]

        cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)
        document_chunks = text_to_docs(cleaned_text_pdf, metadata) # cleaned documents

        # saving each pdf in the form of vectordb
        vectordb = Chroma.from_documents(documents=document_chunks, embedding=embedding,
                                         persist_directory=persist_directory + file.filename)
        vectordb.persist()
        message.append('File stored as vectors successfully')
        return True
    except() as e:
        message.append(e)
        return False

