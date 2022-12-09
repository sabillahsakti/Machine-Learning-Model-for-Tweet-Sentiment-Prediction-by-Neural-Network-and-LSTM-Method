import re
import string
from flask import Flask, jsonify, request #import objects from the Flask model
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
import numpy as np

import pandas as pd
import sqlite3 as sql



app = Flask(__name__) #define app using Flask
app.json_encoder = LazyJSONEncoder

swagger_template = dict(
    info = {
        'title': LazyString(lambda:'API Data Science Platinum Challenge'),
        'version': LazyString(lambda:'1.0.0'),
        'description': LazyString(lambda:'Dokumentasi Data Science Class Platinum Challenge')
        }, host = LazyString(lambda: request.host)
)

swagger_config = {
        "headers":[],
        "specs":[
            {
            "endpoint":'docs',
            "route":'/docs.json'
            }
        ],
        "static_url_path":"/flasgger_static",
        "swagger_ui":True,
        "specs_route":"/docs/"
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)

# READ CSV FILES
# ===============================================================================
def importFileCsv(val):
    # input_file = pd.read_csv(val, encoding='latin-1')
    input_file = pd.read_csv(val, sep="\t", names=["text", "label"])
    word_list = input_file['text'].tolist()
    for j in word_list:
        a = word_list.index(j)
        j = str(j).lower()
        j = re.sub("[,]", "", j)
        j = re.sub("[.]", "", j)
        j = re.sub("[]]", "", j)
        j = re.sub("[[]", "", j)
        j = re.sub("[?]", "", j)
        j = re.sub("[!]", "", j)
        j = re.sub("[\"]", "", j)
        j = re.sub("[']", "", j)
        j = re.sub("[;]", "", j)
        j = re.sub("[_]", "", j)
        j = re.sub("[||]", "", j)
        j = re.sub("[+]", "", j)
        j = re.sub("[#]", "", j)
        j = re.sub("[(]", "", j)
        j = re.sub("[)]", "", j)
        j = re.sub("[ð]", "", j)
        j = re.sub(r'^\s*(-\s*)?|(\s*-)?\s*$', '', j)
        word_list[a] = j
    return word_list

# ===============================================================================
# MY SQL FUNCTION

def inputToTable(val):
    _database = sql.connect('plat.db')
    _database.execute(''' insert into data (teks) values (?) ''', (val,))
    _database.commit()
    _database.close()

def inputListTable(val):
    for i in val:
        inputToTable(i)


# CLEANSING DATA  FILES
# ===============================================================================
def cleansingData(val):
    return val

#SWAGGER UI CODE
#===============================================================================

#POST METHOD MODEL NEURAL NETWORK
@swag_from("docs/post_text_nn.yml", methods=['POST'])
@app.route('/post_text_nn', methods=['POST'])
def postTextNN():
    text = request.form.get('text')
    inputToTable(text)
    return jsonify({'text ' : text})


@swag_from("docs/post_file_nn.yml", methods=['POST'])
@app.route('/post_file_nn', methods=['POST'])
def postFileNN():
    file = request.files['file']
    input_file = importFileCsv(file)
    inputListTable(input_file)
    return jsonify({'text file uploaded ' : input_file.toList})

#POST METHOD MODEL LSTM
@swag_from("docs/post_text_lstm.yml", methods=['POST'])
@app.route('/post_text_lstm', methods=['POST'])
def postTextLSTM():
    text = request.form.get('text')
    inputToTable(text)
    return jsonify({'text ' : text})

@swag_from("docs/post_file_lstm.yml", methods=['POST'])
@app.route('/post_file_lstm', methods=['POST'])
def postFileLSTM():
    file = request.files['file']
    input_file = importFileCsv(file)
    inputListTable(input_file)
    return jsonify({'text file uploaded ' : input_file})


#===============================================================================
if __name__ == '__main__':
    app.run() #run app on port 8080 in debug mode