import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

dataset_path = "./newdrugdataset.xlsx"
data = pd.read_excel(dataset_path)
print("Number of NaN values in the dataset:")
print(data.isnull().sum())

# Feature selection
X = data[["Mass", "M+proton"]]  # Input features
y = data["Compound Name"]  # Target variable

formulas_data = data[["Compound Name", "Formula", "Mass"]]
formulas_data_avg_mass = formulas_data.groupby("Compound Name")["Mass"].mean().reset_index()
formulas_data = pd.merge(formulas_data_avg_mass, formulas_data.drop("Mass", axis=1), on="Compound Name")
formulas_data.set_index("Compound Name", inplace=True)
formulas_data.reset_index(inplace=True)
formulas = formulas_data.to_dict(orient="index")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

model_filename = "molecular_prediction_model.pkl"
formulas_filename = "compound_formulas.pkl"
joblib.dump(model, model_filename)
joblib.dump(formulas, formulas_filename)
print(f"Model saved to {model_filename}")
print(f"Formulas mapping saved to {formulas_filename}")
