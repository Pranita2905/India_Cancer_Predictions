from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
with open("DecisionTree.pkl", "rb") as f:
    model = pickle.load(f)

html = """
<!DOCTYPE html>
<html>
<head>
<title>Cancer Survival Prediction</title>
<meta name="viewport" content="width=device-width, initial-scale=1">

<style>

body{
font-family:Arial,sans-serif;
background:linear-gradient(135deg,#0f172a,#1e40af);
padding:20px;
}

.container{
max-width:1000px;
margin:auto;
background:white;
padding:30px;
border-radius:20px;
box-shadow:0 10px 30px rgba(0,0,0,0.3);
}

h1{
text-align:center;
color:#1e40af;
}

.grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:15px;
}

input,select{
width:100%;
padding:12px;
border:1px solid #ccc;
border-radius:10px;
}

button{
width:100%;
padding:15px;
background:#2563eb;
color:white;
border:none;
border-radius:10px;
font-size:18px;
margin-top:20px;
cursor:pointer;
}

button:hover{
background:#1d4ed8;
}

.result{
margin-top:20px;
padding:15px;
background:#eff6ff;
border-radius:10px;
text-align:center;
font-size:22px;
font-weight:bold;
}

</style>
</head>

<body>

<div class="container">

<h1>🩺 Cancer Survival Prediction System</h1>

<form method="POST">

<div class="grid">

<input type="number" name="Age" placeholder="Age" required>

<input type="number" name="Gender" placeholder="Gender Encoded Value" required>

<input type="number" name="State" placeholder="State Encoded Value" required>

<input type="number" name="City" placeholder="City Encoded Value" required>

<input type="number" name="Cancer_Type" placeholder="Cancer Type Encoded Value" required>

<input type="number" name="Stage" placeholder="Stage Encoded Value" required>

<input type="number" name="Treatment_Type" placeholder="Treatment Type Encoded Value" required>

<input type="number" name="Survival_Months" placeholder="Survival Months" required>

</div>

<button type="submit">Predict</button>

</form>

{% if prediction %}
<div class="result">
{{ prediction }}
</div>
{% endif %}

</div>

</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():

    prediction = ""

    if request.method == "POST":

        features = np.array([[
            float(request.form["Age"]),
            float(request.form["Gender"]),
            float(request.form["State"]),
            float(request.form["City"]),
            float(request.form["Cancer_Type"]),
            float(request.form["Stage"]),
            float(request.form["Treatment_Type"]),
            float(request.form["Survival_Months"])
        ]])

        result = model.predict(features)[0]

        prediction = f"Prediction Result: {result}"

    return render_template_string(html,prediction=prediction)

if __name__ == "__main__":
    app.run()
