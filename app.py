from flask import Flask, request, send_file, render_template, redirect, url_for, session
from fpdf import FPDF
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
import numpy as np
import base64
import json
import struct
import io
import time
from collections import defaultdict
from flask_session import Session
from models import db, User

app = Flask(__name__)

app.secret_key = "drugdetection"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
db = SQLAlchemy(app)
Session(app)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

ALLOWED_EXTENSIONS = {'mzml'}

with app.app_context():
    db.create_all()


def allowed_file(file):
    filename=file.filename
    print(filename.rsplit('.', 1)[1])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def decode_base64_and_unpack(binary_string):
    decoded_bytes = base64.b64decode(binary_string)
    return json.loads(decoded_bytes)


def load_model_and_formulas():
    model_path = "./data/molecular_prediction_model.pkl"
    formulas_path = "./data/compound_formulas.pkl"
    model = joblib.load(model_path)
    formulas = joblib.load(formulas_path)
    return model, formulas

def is_similar_mass(mass1, mass2, threshold_percentage=98):
    try:
        mass1_float = mass1  # Extract numeric part of formula for mass
        mass2_float = mass2
        return abs(mass1_float - mass2_float) / max(mass1_float, mass2_float) * 100 <= (100 - threshold_percentage)
    except ValueError:
        return False

model, formulas = load_model_and_formulas()
unique_compounds = defaultdict(list)
for key, value in formulas.items():
    compound_name = value['Compound Name']
    mass = value['Mass']
    formula = value['Formula']  # Assuming 'Formula' is the key for the formula in your data
    unique_compounds[compound_name].append((mass, formula))  # Store mass and formula as a tuple

unique_formulas = [{'Compound Name': name, 'Mass': sum(mass_formula[0] for mass_formula in masses) / len(masses),
                    'Formula': masses[0][1]} for name, masses in unique_compounds.items()]


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session["logged_in"] = True
            return redirect(url_for("upload_file"))
        else:
            error_message = "Invalid username or password"
            return render_template("login.html", error=error_message)
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if User.query.filter_by(username=username).first():
            error_message = "Username already exists"
            return render_template("signup.html", error=error_message)
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/", methods=["GET"])
def upload_file():
    if "logged_in" in session and session["logged_in"]:
        return render_template("upload.html")
    else:
        return redirect(url_for("login"))

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    if file and allowed_file(file):
        contents = file.read().decode("utf-8")
        start = contents.find("<binary>") + 8
        end = contents.find("</binary>", start)
        encoded_mz = contents[start:end].strip()
        ob_data = decode_base64_and_unpack(encoded_mz)
        print("Obtained data is : ", ob_data)
        mz_values = ob_data["mz_values"]
        print("Mass is : ", mz_values)
        data_for_prediction = pd.DataFrame(
            {
                "Mass": mz_values,
                "M+proton": ob_data["M+proton"],
            }
        )

        prediction = model.predict(data_for_prediction)
        print("prediction is :",prediction)

        prediction_formulas = []
        for compound in unique_formulas:
            if compound['Compound Name'] in prediction:
                prediction_formulas.append(compound['Mass'])
        print("Prediction Formulas is : ", prediction_formulas )

        concentrations = []
        for y, curve, const in zip(ob_data["M+peak"], ob_data["calibration curve"], ob_data["calibration constant"]):
            print(y," ",const," ",curve)
            concentrations.append(f"{(y-const)/curve:.3f}")
        print("Concentrations are : ", concentrations)


        filtered_predictions = []
        for name, formula, mz, mass, concentration in zip(prediction, unique_formulas, mz_values, prediction_formulas, concentrations):
            try:
                compound_formula = next((f for f in unique_formulas if f["Compound Name"] == name), None) 
                if is_similar_mass(mz, mass):
                    filtered_predictions.append((name, compound_formula["Formula"], mz, concentration))
            except ValueError:
                name = "Unknown"
                formula = "Unknown"
                filtered_predictions.append((name, formula, mz, concentration))
        time.sleep(2)
        print("Filtered Prediction is : ", filtered_predictions)

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Prediction Results", ln=True, align="C")
        # Add table header
        if filtered_predictions != []:
            pdf.cell(60, 10, "Drug Name", border=1)
            pdf.cell(40, 10, "Formula", border=1)
            pdf.cell(40, 10, "M/Z Value", border=1)
            pdf.cell(50, 10, "Concentration (µg/mL)", border=1, ln=True)

            for name, formula, mz, concentration in filtered_predictions:
                pdf.cell(60, 10, name, border=1)
                pdf.cell(40, 10, formula, border=1)
                pdf.cell(40, 10, str(mz), border=1)
                pdf.cell(50, 10, str(concentration), border=1, ln=True)


            pdf_filename = "temp_prediction_results.pdf"
            pdf.output(pdf_filename)

            session["results"] = [
                {"Drug Name": name, "Formula": formula, "M/Z Value": mz, "Concentration":concentration}
                for name, formula, mz, concentration in filtered_predictions
            ]
        else:
            pdf.cell(100,10,"No Drug Found",border=1, ln=True, align="C")
            pdf_filename = "temp_no_drug_found.pdf"
            pdf.output(pdf_filename)
            session["results"] = []
        return redirect(url_for("results", filename=pdf_filename))
    else:
        error_message = "Please upload a valid .mzML file."
        return render_template("upload.html", error=error_message)


@app.route("/results")
def results():
    results = session.get("results", [])
    filename = request.args.get("filename")
    return render_template("results.html", filename=filename, results=results)


@app.route("/download/<filename>")
def download(filename):
    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
