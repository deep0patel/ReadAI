# ReadAI

## Description

This is a python web application. You can load multiple books in the form of pdf. This books will be split and stored in chunks. This chunks are converted to Embeddings(high dimensional vectors) which will be stored locally. Spliiting and  storing takes time depending on your device. 

### How to install and run this app.

1. Clone the repository:

```bash
git clone https://github.com/deep0patel/ReadAI/
```
2. Navigate:

```bash
cd ReadAI
```

3. Install the required dependencies using pip command:

```bash
pip install -r requirements.txt
```

4. Run project

```bash
python run main.py
```

### Guide

As this is a web app you will have to access it on browser. This app is running on 8000 port. So type below link to start with the first page.

http://localhost:8000/key

On the first page you will be asked to enter your openAI api key. Get it from openAI's website and enter it. This key will be stored as session variable and will be erased as soon as the application closes. So, you will have to keep it in handy to use the application again.

Now you can select or upload pdf. While uploading be patient as it take some time to process if pdf is big. once done you willl be redirected to question answer page.

Thanks.

