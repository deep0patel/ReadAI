import os.path
import os
from flask_wtf import FlaskForm
from flask import Flask, request, render_template, Blueprint, url_for, flash, session
from werkzeug.utils import secure_filename, redirect
from wtforms import FileField, SubmitField
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .pdfProcessor2 import *

views = Blueprint('views', __name__)


class HomePageForm(FlaskForm):
    select_button = SubmitField('Select')
    upload_button = SubmitField('Upload')


class UploadFileForm(FlaskForm):
    file = FileField('File')
    submit = SubmitField('Upload')


class ChatFileForm(FlaskForm):
    submit = SubmitField('Process')


# @views.route('/', methods=['GET', 'POST'])
# def home():
#     form = homepageForm()
#     if form.validate_on_submit():
#         if form.select_button.data:
#             return redirect(url_for('views.home'))
#         elif form.upload_button.data:
#             return redirect(url_for('views.upload_pdf'))
#
#     return render_template('homepage.html', form=form)

@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        message = []
        if 'select' in request.form:
            return redirect(url_for('views.select_pdf'))
        elif 'upload' in request.form:
            return redirect(url_for('views.upload_pdf'))
    return render_template('homepage.html')


@views.route('/select-pdf', methods=['GET', 'POST'])
def select_pdf():
    if request.method == 'POST':
        if 'select' in request.form:
            selected_option = request.form.get('selected_option')
            if selected_option:
                session['book_name'] = selected_option
                # Process the selected option
                # You can add your logic here based on the selected option
                # For example, you can store it in a database or perform some action

                # Redirect to a different page with a success message
                return redirect(url_for('views.chat', option=selected_option))

    message = []
    # Load all the pdf names in the form of a list
    pdf_list = get_pdf_list(message)
    if len(pdf_list) == 0:
        message.append('No pdf found. Please upload.')
        flash(message)
        return redirect(url_for('views.upload_pdf'))

    return render_template('select_pdf.html', options=pdf_list)


@views.route('/upload-pdf', methods=['GET', 'POST'])
def upload_pdf():
    form = UploadFileForm()
    message = []
    if form.validate_on_submit():
        file = form.file.data  # fetching pdf from form

        success = pdf_upload(file, message)  # saving file
        if success:
            session['book_name'] = file.filename
            return redirect(url_for('views.chat'))

    return render_template('upload_pdf.html', message=message, form=form)


@views.route('/process', methods=['GET', 'POST'])
def process():
    message = []
    if 'process' in request.form:
        if pdf_process(message):
            return redirect(url_for('views.process'))
        else:
            return render_template('upload_success.html', message=','.join(message))
    if 'embedding' in request.form:
        if csv_to_emb(message):
            return redirect(url_for('views.chat'))
        else:
            return render_template('upload_success.html', message=','.join(message))
    return render_template('upload_success.html')


@views.route('/chat', methods=['GET', 'POST'])
def chat():
    return render_template('chat.html')


@views.route('/query', methods=['GET', 'POST'])
def query():
    message = []
    if request.method == 'POST':
        if 'answer' in request.form:
            query_text = request.form.get('query')
            is_empty = not bool(query_text)

            if is_empty:
                message.append('Enter Question to continue.')
                return render_template('chat.html', message=message)
            message = answer(query_text)
        elif 'original_content' in request.form:
            message = 'You clicked original_content 2!'
    return render_template('chat.html', message=message)


@views.route('/key', methods=['GET', 'POST'])
def key():
    message = []
    if request.method == 'POST':
        if 'submit' in request.form:
            api_key = request.form.get('key')
            is_empty = not bool(api_key)
            if is_empty:
                message.append('Enter api key to continue.')
                return render_template('api_key.html', message=message)

            is_key_valid = check_api_key(api_key, message)

            if is_key_valid:
                session['api_key'] = api_key
                return redirect(url_for('views.home'))
            else:
                message.append('Api key not valid ! Try again.')
                return render_template('api_key.html', message=message)

    return render_template('api_key.html', message=message)

# make it clear how exactly from submission works.
# may be we can use session variable to save openai api keys.
# talk to someone about your next project to work with you.

# You could POST an html form to your /select endpoint
#
# <form method="post" action="/select">
#   <select id="operator" name="operator">
#       <option value="=">=</option>
#       <option value=">">></option>
#       <option value="<"><</option>
#   </select>
#   <input type="submit" value="Submit">
# </form>
# Then the value could be referenced using request.form['operator']
