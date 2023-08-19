import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
import nltk
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report



st.title('Bangla Review Sentiment Analysis')
st.write('''
         Please fill in the Box below, then hit the Predict button
         to get your results. Depending on the Model it may take 0-4 Minutes to Generate Results
         ''')
st.write('''Note: MLPClassifier Models Gives the Best Results''')

txt = st.text_area('Text to analyze', '''
    ''')

selected_models = st.multiselect("Choose Classifier Models", (
    'Random Forest', 'Naïve Bayes', 'Logistic Regression',
    'K-Nearest Neighbors', 'Decision Tree', 'Gradient Boosting', 'Extra Trees', 'SVC',
    'MLPClassifier', 'XGBClassifier'
))

st.write(''' ''')

# Initialize an empty list to store the selected models
models_to_run = []

# Initialize classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Naïve Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'SVC': SVC(),
    'MLPClassifier': MLPClassifier(),
    'XGBClassifier': XGBClassifier()
}

user_input = txt

# Load dataset
def get_dataset():
    data = pd.read_csv('data.csv')
    data.columns = ["id", "date", "name", "rating", "sentiment", "comment"]
    return data

def generate_model_labels(model_names):
    model_labels = []
    for name in model_names:
        words = name.split()
        label = "".join(word[0] for word in words)
        model_labels.append(label)
    return model_labels

if st.button('Submit'):
    df = get_dataset()
    nltk.download('punkt')

    # Data Preprocessing
    df = df[['comment', 'sentiment']]
    df = df.drop_duplicates()
    df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
    df = df.dropna()

    # Tokenization
    df['tokenized_comment'] = df['comment'].apply(word_tokenize)

    # Data Augmentation - Oversampling
    oversampler = RandomOverSampler(random_state=101)
    X_over, y_over = oversampler.fit_resample(df['tokenized_comment'].to_frame(), df['sentiment'])

    # Data Augmentation - Undersampling
    undersampler = RandomUnderSampler(random_state=1337)
    X_augmented, y_augmented = undersampler.fit_resample(X_over, y_over)

    # Feature Selection using TF-IDF with Trigram Features
    X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=1337)

    # Initialize the TF-IDF vectorizer with trigram features
    tfidf = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False, ngram_range=(1, 3))

    # Fit and transform the training data
    X_train_tfidf = tfidf.fit_transform(X_train['tokenized_comment'])
    X_test_tfidf = tfidf.transform(X_test['tokenized_comment'])

    results = []

    for model_name in selected_models:
        model = classifiers[model_name]
        model.fit(X_train_tfidf, y_train)
        model_predictions = model.predict(X_test_tfidf)

        model_accuracy = accuracy_score(y_test, model_predictions)
        model_precision = precision_score(y_test, model_predictions)
        model_recall = recall_score(y_test, model_predictions)
        model_f1score = f1_score(y_test, model_predictions)

        sentiment_prediction = 'Positive' if model_predictions[0] == 1 else 'Negative'

        st.write(f'According to {model_name} Model Sentiment is {sentiment_prediction}.')
        st.write(f'{model_name} Accuracy:', model_accuracy)
        st.write(f'{model_name} Precision:', model_precision)
        st.write(f'{model_name} Recall:', model_recall)
        st.write(f'{model_name} F1 Score:', model_f1score)
        st.write(
            '------------------------------------------------------------------------------------------------------')

        results.append({
            'Model': model_name,
            'Accuracy': model_accuracy,
            'Precision': model_precision,
            'Recall': model_recall,
            'F1 Score': model_f1score
        })

        results_df = pd.DataFrame(results)

    # Shorten model names for plotting
    results_df['Short Model'] = generate_model_labels(results_df['Model'])

    # Plotting
    plt.figure(figsize=(12, 8))

    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    bar_colors = ['skyblue', 'orange', 'green', 'purple']

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.bar(results_df['Short Model'], results_df[metric], color= bar_colors[i])
        plt.title(f'{metric} Comparison')
        plt.ylim(0, 1)

    plt.tight_layout()
    st.pyplot()