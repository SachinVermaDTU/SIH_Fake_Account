# coding: utf-8

### Detect fake profiles on Instagram using Random Forest

import sys
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve

####### function for reading dataset from csv files

def read_datasets():
    """ Reads Instagram profile data from CSV files """
    real_profiles = pd.read_csv("data/real_profiles.csv")
    fake_profiles = pd.read_csv("data/fake_profiles.csv")
    
    # Combine real and fake profiles
    x = pd.concat([real_profiles, fake_profiles])   
    # Label: 0 for fake, 1 for real
    y = len(fake_profiles) * [0] + len(real_profiles) * [1]
    return x, y


####### function for extracting features

def extract_features(x):
    """ Extract key features from the Instagram profiles """
    # Convert categorical/binary features to numerical values
    x['has_pic'] = x['has_pic'].apply(lambda p: 1 if p == 'Yes' else 0)
    x['is_business'] = x['is_business'].apply(lambda b: 1 if b == 'Yes' else 0)
    x['is_private'] = x['is_private'].apply(lambda p: 1 if p == 'Yes' else 0)
    x['is_verified'] = x['is_verified'].apply(lambda v: 1 if v == 'Yes' else 0)
    x['has_channel'] = x['has_channel'].apply(lambda c: 1 if c == 'Yes' else 0)
    x['connected_fb_page'] = x['connected_fb_page'].apply(lambda f: 1 if f == 'Yes' else 0)

    # Select relevant features for the model
    feature_columns_to_use = ['UName', 'Fullname', 'has_pic', 'biography', 'Followedby',
                              'Followed', 'Is_Followed_More', 'Postcount', 'is_business', 
                              'is_private', 'is_verified', 'has_channel', 'external_url', 
                              'highlight_reel_count', 'connected_fb_page']
    x = x.loc[:, feature_columns_to_use]
    return x


####### function for plotting learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    # Plot learning curve
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()


####### function for plotting confusion matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names = ['Fake', 'Genuine']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


####### function for plotting ROC curve

def plot_roc_curve(y_test, y_pred):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


####### function for training data using Random Forest

def train(X_train, y_train, X_test):
    """ Trains and predicts dataset with a Random Forest classifier """
    
    clf = RandomForestClassifier(n_estimators=40, oob_score=True)
    clf.fit(X_train, y_train)
    print("The best classifier is: ", clf)
    
    # Estimate score using cross-validation
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    
    # Plot learning curve
    plot_learning_curve(clf, 'Learning Curves (Random Forest)', X_train, y_train, cv=5)
    
    # Predict
    y_pred = clf.predict(X_test)
    return y_test, y_pred


####### Main program execution

if __name__ == '__main__':
    print("Reading datasets...\n")
    x, y = read_datasets()
    print(x.describe())
    
    print("Extracting features...\n")
    x = extract_features(x)
    print(x.columns)
    print(x.describe())
    
    print("Splitting datasets into train and test sets...\n")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=44)
    
    print("Training datasets...\n")
    y_test, y_pred = train(X_train, y_train, X_test)
    
    # Print classification accuracy
    print('Classification Accuracy on Test dataset:', accuracy_score(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix, without normalization')
    print(cm)
    plot_confusion_matrix(cm)
    
    # Plot normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred)
