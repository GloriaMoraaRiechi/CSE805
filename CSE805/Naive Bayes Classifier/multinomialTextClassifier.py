from idlelib.iomenu import encoding

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud

sms_data = pd.read_csv(r"C:\Users\glori\Desktop\CSE805\CSE805\Naive Bayes Classifier\spam.csv", encoding='latin-1')
print(sms_data.head())

# Preprocess the data
sms_data = sms_data[['v1', 'v2']]
sms_data = sms_data.rename(columns={'v1': 'label', 'v2': 'text'})

# Split the data into features and labels
X = sms_data['text']
y = sms_data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exploratory Data Analysis: helps us gain deeper understanding about the dataset
# EDA 1: Distribution of Classes
# Distribution of classes: Visualizing distribution of the classes of target variable helps us to understand its potential behavior. Here we will generate a pie-chart for both classes(‘spam’ and ‘ham’) of target variable.
class_distribution = sms_data['label'].value_counts()
class_distribution.plot(kind='pie', autopct='%1.1f%%', colors=['#66b3ff','#99ff99'])
plt.title('Distribution of Spam and Ham Messages')
plt.show()

# Generating Word-cloud:
# As our dataset contains text dataset so that generating word clouds for relevant words in spam and ham messages provides a visual representation of the most common and relevant terms.
# Generate WordCloud for Spam Messages
spam_text = ' '.join(sms_data[sms_data['label'] == 'spam']['text'])
spam_wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white', random_state=42).generate(
    spam_text)

# Generate WordCloud for Ham Messages
ham_text = ' '.join(sms_data[sms_data['label'] == 'ham']['text'])
ham_wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white', random_state=42).generate(
    ham_text)

# Plot the WordClouds
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Spam Messages')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(ham_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Ham Messages')
plt.axis('off')

plt.tight_layout()
plt.show()

# Model Training
# Before training models we need to vectorize the text data to convert it to numerical dataset. To do this we will use Count Vectorizer. After that we will train Multinomial NB and also Gaussian NB to show comparative performance.
# Create a CountVectorizer to convert text data into numerical features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
mnb = MultinomialNB(alpha=0.8, fit_prior=True, force_alpha=True)
mnb.fit(X_train_vec, y_train)

# Train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train_vec.toarray(), y_train)

# Model Evaluation
# Evaluate the models using accuracy and F1-score
y_pred_mnb = mnb.predict(X_test_vec)
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
f1_mnb = f1_score(y_test, y_pred_mnb, pos_label='spam')

y_pred_gnb = gnb.predict(X_test_vec.toarray())
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
f1_gnb = f1_score(y_test, y_pred_gnb, pos_label='spam')

# Print the results
print("Multinomial Naive Bayes - Accuracy:", accuracy_mnb)
print("Multinomial Naive Bayes - F1-score for 'spam' class:", f1_mnb)

print("Gaussian Naive Bayes - Accuracy:", accuracy_gnb)
print("Gaussian Naive Bayes - F1-score for 'spam' class:", f1_gnb)

# Comparative visualization
methods = ['Multinomial Naive Bayes', 'Gaussian Naive Bayes']
accuracy_scores = [accuracy_mnb, accuracy_gnb]
f1_scores = [f1_mnb, f1_gnb]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(methods, accuracy_scores, color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')

plt.subplot(1, 2, 2)
plt.bar(methods, f1_scores, color=['blue', 'green'])
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison')

plt.show()

# Conclusion
# We can conclude that MNB is very efficient algorithm for NLP based tasks. Here MNB achieves a notable 98% of accuracy and 93.70% of F1-score which is far better than GNB where F1-score is less.

