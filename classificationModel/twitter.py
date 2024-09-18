import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout, SpatialDropout1D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Download NLTK data if not already installed
nltk.download('punkt')

####### Function to read the dataset
def read_dataset(file_path):
    """Reads the dataset from a CSV file."""
    data = pd.read_csv(file_path)  # Provide the correct path to your dataset
    return data

####### Function to clean and preprocess text
def clean_text(text):
    """Cleans text by removing mentions, hashtags, URLs, special characters, and numbers."""
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove extra spaces
    return text

####### Function to extract features
def extract_features(data):
    """Extracts and preprocesses features using tokenization and dimensionality reduction."""
    # Clean text
    data['cleaned_text'] = data['tweet'].apply(clean_text)

    # Tokenization
    data['tokenized_text'] = data['cleaned_text'].apply(word_tokenize)

    # Feature extraction using CountVectorizer (Bag of Words model)
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['cleaned_text']).toarray()

    # Dimensionality reduction using PCA
    pca = PCA(n_components=500)  # Reduce to 500 components
    X_pca = pca.fit_transform(X)

    # Label encoding for the target variable
    le = LabelEncoder()
    y = le.fit_transform(data['account_type'])  # Assume 'account_type' has 'fake' and 'legitimate' labels

    return X_pca, y, data

####### Function to prepare data for LSTM-CNN
def prepare_data_for_lstm(data, X_train, X_test):
    """Prepares data for LSTM by tokenizing and padding sequences."""
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data['cleaned_text'])

    X_train_seq = tokenizer.texts_to_sequences(data.loc[X_train.index, 'cleaned_text'])
    X_test_seq = tokenizer.texts_to_sequences(data.loc[X_test.index, 'cleaned_text'])

    # Padding sequences to ensure equal input size
    max_seq_length = 100  # Adjust based on your data
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_length)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_length)

    return X_train_pad, X_test_pad, tokenizer

####### Function to build the LSTM-CNN model
def build_model(max_seq_length, embedding_dim=128):
    """Builds the LSTM-CNN hybrid model."""
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=embedding_dim, input_length=max_seq_length))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification: fake (1) or legitimate (0)
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

####### Function to train the model
def train_model(model, X_train_pad, y_train, X_test_pad, y_test):
    """Trains the model and returns the training history."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test), epochs=10, batch_size=64, callbacks=[early_stopping])
    return history

####### Function to evaluate the model
def evaluate_model(model, X_test_pad, y_test):
    """Evaluates the model's performance and plots accuracy and loss."""
    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    
    # Plot accuracy and loss
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(model.history.history['accuracy'], label='Training Accuracy')
    plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(model.history.history['loss'], label='Training Loss')
    plt.plot(model.history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

####### Main Program
if __name__ == '__main__':
    # Read dataset
    print("Reading dataset...")
    data = read_dataset('mib_dataset.csv')  # Replace with the correct path to your dataset
    
    # Extract features
    print("Extracting features...")
    X_pca, y, data = extract_features(data)
    
    # Train-Test split
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # Prepare data for LSTM
    print("Preparing data for LSTM...")
    X_train_pad, X_test_pad, tokenizer = prepare_data_for_lstm(data, X_train, X_test)
    
    # Build the model
    print("Building the model...")
    model = build_model(max_seq_length=100)
    
    # Train the model
    print("Training the model...")
    history = train_model(model, X_train_pad, y_train, X_test_pad, y_test)
    
    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test_pad, y_test)
