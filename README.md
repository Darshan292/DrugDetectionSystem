# Drug Detection Web Application

This project is a web application for detecting drugs using a machine learning model trained on mass spectrometry data from mzML files. It predicts the compound name of a drug if detected in the uploaded mzML file; otherwise, it indicates that the report is normal. The application also supports user authentication and generates PDF reports for the predictions.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.6+
- Flask
- SQLAlchemy
- scikit-learn
- pandas
- joblib
- fpdf
- Flask-Session


## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/drug-detection-web-app.git
cd drug-detection-web-app
```
2. **Install the required Packages:**

```bash
pip install -r requirements.txt
```
3. **Set up the SQLite database:**

```bash
python -c "from main import db; db.create_all()"
```
4. **Place the trained model and formulas files in the data directory:**
- molecular_prediction_model.pkl
- compound_formulas.pkl

## Usage
1. **Run the Flask application:**
```
python main.py
```
2. **Open your web browser and navigate to:**
``` arduino
http://127.0.0.1:5000/
```
3. **Sign up or log in to access the file upload and prediction features.**

## Model Training
The model was trained using a dataset of drug compounds with their mass and M+proton values. The script for training the model includes reading the dataset, feature selection, splitting the data into training and testing sets, training a RandomForestClassifier, evaluating the model, and saving the trained model and formulas to files.

## Main Application Code
The main code for the Flask application handles user authentication, file upload, drug detection, and PDF report generation. It includes routes for login, signup, file upload, prediction, displaying results, and downloading the PDF report.

