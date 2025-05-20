import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("customer_churn.csv")  # Replace with your dataset path

# Basic preprocessing
df = df.dropna()  # Remove missing values
le = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = le.fit_transform(df[column])

# Feature/Target split
X = df.drop("Churn", axis=1)  # 'Churn' should be the target column
y = df["Churn"]

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
features = df.drop("Churn", axis=1).columns
sns.barplot(x=importances, y=features)
plt.title("Feature Importance in Churn Prediction")
plt.tight_layout()
plt.show()
