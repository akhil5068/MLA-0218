import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

# Load your dataset
df = pd.read_csv("iris.csv")  # Ensure this CSV has correct column names

# Check column names and update accordingly
print(df.head())  # Uncomment to check the first few rows

# Define features (X) and target (y)
X = df.drop(columns=['species'])  # Replace 'species' with the correct column name for labels
y = df['species']  # Replace with actual target column name

# Encode categorical target labels (if necessary)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Converts 'setosa', 'versicolor', etc. into numeric labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
