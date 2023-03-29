from flask import Flask, render_template, url_for, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np
import spacy
import re

app = Flask(__name__)

#load language model for preprocessing
nlp = spacy.load("en_core_web_sm")
#load the model
model = load_model("./assets/movieSentiAnalysisV-4.h5")

#stopwords
stop_word_text = ['a', 'an', 'br', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'who', 
                  'whom', 'which', 'that', 'this', 'these', 'those', 'in', 'on', 'at', 'to', 'from', 'by', 'for', 'of', 'with',
                  'without', 'over', 'under', 'above', 'below', 'between', 'among', 'through', 'throughout', 'until', 'while', 
                  'since', 'during', 'within', 'without', 'beyond', 'beside', 'between', 'except', 'but', 'up', 'down', 'in', 
                  'out', 'off', 'above', 'below', 'under', 'too', 'very', 'so', 'such', 'just', 'as', 'both', 'neither', 'either', 
                  'although', 'because', 'since', 'so that', 'though', 'this', 'I', 'i', 'she', 'he', 'they', 'it', 'unless', 
                  'until', 'whether', 'while', 'why', '<', '>', 'it', 'that']

#filtering the text
def full_form(text):
    text = text.lower()
    plain = re.sub(r'[<>?\.,!"(\)\/[\]]', '', text)
    plain = plain.replace("don't", "do not")
    plain = plain.replace("won't", "will not")
    plain = plain.replace("haven't", "have not")
    plain = plain.replace("can't", "cannot")
    plain = plain.replace("she's", "she is")
    plain = plain.replace("he's", "he is")
    plain = plain.replace("there're", "there are")
    plain = plain.replace("they'd", "they would")
    plain = plain.replace("\'ll", " will")
    return plain    

#preprocessing  text 
def preprocess_text_data(data):
    corpus = []
  
    #split the sentence
    plain = full_form(data)
    #stemming
    doc = nlp(plain)
    # Apply stemming and remove stopwords
    stemmed_text = []
    for token in doc:
        stemmed_text.append(token.lemma_)
    
    stemmed_text = [word for word in stemmed_text if word.lower() not in stop_word_text]
    
    #rejoining text
    preprocessed_text = ' '.join(stemmed_text)
    #add the sentence into list
    corpus.append(preprocessed_text)
    return corpus

voc_size = 5000
sent_length = 200

@app.route('/', methods=['POST', 'GET'])
def index():
    text = ""
    result = ""
    #get the user input 
    if request.method == "POST":
        text = request.form['user-input']
    #preprocessing 
    processed_text = preprocess_text_data(text)
    #one hot encoding
    onehot_sent = [one_hot(word, voc_size) for word in processed_text]
    #embedding
    embedd_docs = pad_sequences(onehot_sent, padding='pre', maxlen=sent_length)
    #convert into numpy array
    X_sample = np.array(embedd_docs)
    
    prediction = (model.predict(X_sample) > 0.9).astype("int32")
    if prediction[0][0] == 0:
        result = "Negative \U0001F614"
    elif prediction[0][0] == 1:
        result = "Positive \U0001F603" 
    return render_template('index.html', text=result)



if __name__ == "__main__":
    app.run(debug=True)