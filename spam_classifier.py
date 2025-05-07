
# importing packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv("data/emails.csv")

# checking data 
# print(df)
# print(df.groupby('Spam/Ham').describe())

# Just keeping the message body and whether it's spam or not
df = df[['Message', 'Spam/Ham']]
df.columns = ['text', 'label']

# Clean up any  spacing and make sure all values are usable
df['label'] = df['label'].str.strip().str.lower()
df['text'] = df['text'].astype(str)
df = df[df['text'].str.strip() != '']  # drop  empty messages

# Features (x-email messages) and target labels (y-spam/ham)
X = df['text']
y = df['label']

# Split the dataset – 75% for training, 25% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)

# Turn the email text into numbers that the model can understand
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict on the test set
y_pred = model.predict(X_test_vec)

# Pre-test spam
spam_check = ["click link to claim your rewards!"]
spam_check_count = vectorizer.transform(spam_check)
print(model.predict(spam_check_count))

# Pre-test ham
ham_check = ["Hey, are we still on for Friday?"]
ham_check_count = vectorizer.transform(ham_check)
print(model.predict(ham_check_count))


# See how it performed
print("\nHow did the model do?\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nDetailed Breakdown (Precision, Recall, F1):")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Take users input to check their email
while True:
    print("\nEnter an email to classify (or type 'exit' to quit):")
    user_input = input(">> ")
    if user_input.lower() == 'exit':
        break

    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)
    print(f"Prediction: {prediction[0].upper()}")
