%load_ext autoreload
%autoreload 2

# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

file_path = '/content/parkinsons.csv'  # Update with your file path
df = pd.read_csv(file_path)
df = df.dropna()
# Define features (DFA and PPE) and target (status)
X = df[['DFA', 'PPE']]  # Replace with exact column names for HNR and PPE
y = df['status']  # Target column (1 = Parkinson's, 0 = Healthy)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=7)  # Use 5 neighbors
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


joblib.dump(knn, 'my_model.joblib')
