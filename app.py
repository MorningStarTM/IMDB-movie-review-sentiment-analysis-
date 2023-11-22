from flask import Flask, render_template, url_for, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np
import re

app = Flask(__name__)

#load the model
model = load_model("./assets/movieSentiAnalysisV-5.h5")


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


voc_size = 5000
sent_length = 200

@app.route('/', methods=['POST', 'GET'])
def index():
    text = ""
    result = ""
    #get the user input 
    if request.method == "POST":
        text = request.form['user-input']

    if text != "":
        processed_text = full_form(text)
        #one hot encoding
        onehot_sent = []
        for word in processed_text.split(" "):
            # Check if the result of one_hot is non-empty
            onehot_word = one_hot(word, voc_size)
            if onehot_word:
                onehot_sent.append(onehot_word[0])
        #embedding
        embedd_docs = pad_sequences([onehot_sent], padding='pre', maxlen=sent_length)
        #convert into numpy array
        sample = np.array(embedd_docs)
    
        prediction = (model.predict(sample) > 0.9).astype("int32")
        if prediction[0][0] == 0:
            result = "Negative \U0001F614"
        elif prediction[0][0] == 1:
            result = "Positive \U0001F603"
    return render_template('index.html', text=result)



if __name__ == "__main__":
    app.run(debug=True)