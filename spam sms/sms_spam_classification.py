# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load Dataset
data_path = "./sms/spam.csv"  # Adjust path if needed
data = pd.read_csv(data_path, encoding='latin-1')
data = data[['v1', 'v2']]  # Select necessary columns
data.columns = ['label', 'message']  # Rename columns for clarity

# Encode labels: 'ham' -> 0, 'spam' -> 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 2: Data Splitting
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Text Vectorization
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test_vec)

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# Step 6: Prediction Function
def predict_sms(sms):
    sms_vectorized = vectorizer.transform([sms])  # Transform input SMS to vectorized form
    prediction = model.predict(sms_vectorized)[0]
    return "Spam" if prediction == 1 else "Ham"

# Step 7: Test the Prediction Function
sample_sms = "The value of an MCA is significantly enhanced when it comes from a top-tier institution. Attending a highly-ranked Online MCA Program provides a competitive edge, making it easier to secure coveted job offers and achieve career advancements. Introducing SRM Institute of Science and Technology, one of India’s top-ranked institutions renowned for its academic excellenc."
print(f"Message: {sample_sms}")
print(f"Prediction: {predict_sms(sample_sms)}")

sample_sms_2 = "Hi, are we still meeting at 3 PM today?"
print(f"\nMessage: {sample_sms_2}")
print(f"Prediction: {predict_sms(sample_sms_2)}")
