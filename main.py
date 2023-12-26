from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/recommend_books', methods = ["POST"])
def recommend():
    user_input = request.form.get("user_input")
    return str(user_input)


if __name__ == "__main__":
    app.run(debug = True)