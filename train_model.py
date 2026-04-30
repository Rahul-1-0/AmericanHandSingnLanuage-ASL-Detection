import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# load dataset
data = pd.read_csv("dataset.csv", header=None)

X = data.iloc[:, :-1]   # features (landmarks)
y = data.iloc[:, -1]    # labels (A,B,C...)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)

# accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# save model
joblib.dump(model, "sign_model.pkl")

print("Model saved as sign_model.pkl")