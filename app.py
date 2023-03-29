from flask import Flask, render_template, url_for, request

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    text = ""
    if request.method == "POST":
        text = request.form['user-input']

    return render_template('index.html', text=text)



if __name__ == "__main__":
    app.run(debug=True)