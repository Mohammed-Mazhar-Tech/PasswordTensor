from flask import Flask, render_template, request
from password_analyser import predict_strength

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = confidence = explanation = flags = None
    if request.method == 'POST':
        password = request.form['password']
        result, confidence, explanation, flags = predict_strength(password)
    return render_template("index.html", result=result, confidence=confidence, explanation=explanation, flags=flags)

if __name__ == "__main__":
    app.run(debug=True)
