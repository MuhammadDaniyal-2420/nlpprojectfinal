
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace '/content/drive/MyDrive/Phishing_Email.csv' with the actual file path)
file_path = 'Phishing_Email.csv'
df = pd.read_csv(file_path)

# Create a Streamlit app
st.title("Phishing Email Detector")

# Display a sample of the dataset
st.subheader("Sample of the Dataset")
st.dataframe(df.head())

# Sidebar for user input
st.sidebar.title("Choose Classifier and Parameters")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['Email Text'], df['Email Type'], test_size=0.2, random_state=42
)

# Handle missing values
X_train = X_train.fillna("")  # Replace NaN values with an empty string
X_test = X_test.fillna("")

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Choose a classifier
classifier = st.sidebar.selectbox("Select Classifier", ["Naive Bayes"])
if classifier == "Naive Bayes":
    st.sidebar.subheader("Naive Bayes Parameters")

# Train the model
if classifier == "Naive Bayes":
    alpha = st.sidebar.slider("Alpha (Smoothing)", 0.1, 1.0, 1.0, 0.1)
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vectorized)

# Display model performance metrics
st.subheader("Model Performance Metrics")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

# Display classification report
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Phishing email detection function
def detect_phishing_email(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

# Streamlit interface for user input
st.subheader("Enter an Email Text to Check for Phishing")
user_input = st.text_area("Email Text", "")
if st.button("Detect Phishing"):
    if user_input:
        result = detect_phishing_email(user_input)
        st.success(f"The email is classified as: {result}")
    else:
        st.warning("Please enter an email text.")
