import os
import csv
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from numpy import array
from numpy import argmax
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Function to classify using Gaussian NB, Linear SVC, and K-Nearest Neighbors
def classify(list_of_predictors):
    cols = [col for col in df.columns if col in list_of_predictors]
    x = df[cols]
    y = df["genre"]
    # Splitting into x, y, train (80%) and test (20%)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state = 42)
    # GAUSSIAN NAIVE BAYES
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred = gnb.predict(x_test)
    gaussian_accuracy = accuracy_score(y_test,y_pred)*100
    # LINEAR SVC 
    from sklearn.svm import LinearSVC
    svc_model = LinearSVC(random_state=42,dual=False,max_iter = 1500)
    svc_model.fit(x_train, y_train)
    pred = svc_model.fit(x_train, y_train).predict(x_test)
    svc_score = accuracy_score(y_test, pred, normalize = True)*100
    # K-NEAREST NEIGHBORS
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train,y_train)
    pred = neigh.predict(x_test)
    knn_score = accuracy_score(y_test, pred)*100
    # Saving results into a dictionary
    classifier_results = dict(zip(["GNB","Linear SVC","KNN"],[gaussian_accuracy,svc_score,knn_score]))
    return classifier_results

# Function to classify using only N best features
def classify_best_features(list_of_predictors,n):
    cols = [col for col in df.columns if col in list_of_predictors]
    x = df[cols]
    y = df["genre"]
    # Removing features with low variance 
    var = VarianceThreshold(threshold=(0.5))
    new_x = var.fit_transform(x)
    # Selecting N best features
    selector = SelectKBest(f_classif,k=n)
    new_pred = selector.fit_transform(new_x,y)
    # Splitting into x, y, train (80%) and test (20%)
    x_train, x_test, y_train, y_test = train_test_split(new_pred,y, test_size = 0.20, random_state = 42)
    # GAUSSIAN NAIVE BAYES
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred = gnb.predict(x_test)
    gaussian_accuracy = accuracy_score(y_test,y_pred)*100
    # LINEAR SVC 
    from sklearn.svm import LinearSVC
    svc_model = LinearSVC(random_state=42,dual=False,max_iter=1500)
    svc_model.fit(x_train, y_train)
    pred = svc_model.fit(x_train, y_train).predict(x_test)
    svc_score = accuracy_score(y_test, pred, normalize = True)*100
    # K-NEAREST NEIGHBORS
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train,y_train)
    pred = neigh.predict(x_test)
    knn_score = accuracy_score(y_test, pred)*100
    # Saving results into a dictionary
    classifier_results = dict(zip(["GNB","Linear SVC","KNN"],[gaussian_accuracy,svc_score,knn_score]))
    return classifier_results

# Making dataframe out of dictionary
df = pd.read_csv("comb_dataset.csv")
# Let's look at the genre distribution
genre_count  = df['genre'].value_counts()
plt.figure()
sns.barplot(genre_count.index, genre_count.values)
plt.title('Songs per genre in final dataset')
plt.ylabel('Number of songs', fontsize=12)
plt.xlabel('genre', fontsize=12)
plt.savefig('distribution_genre.png')
# Converting categorical outcome variable into numerical
le = preprocessing.LabelEncoder()
df['genre'] = le.fit_transform(df['genre'])

# CLASSIFICATION BASED ON ALL PREDICTORS
all_predictors = [col for col in df.columns if col not in ["genre","track_id","Unnamed: 0"]]
audio_features = [line.strip() for line in open('audio_feature_names.txt')]
lyrics_features = [col for col in all_predictors if col not in audio_features]

all_pred_results = dict(zip(["Audio Features","Lyrics"],[classify(audio_features),classify(lyrics_features)]))
with open('all_pred_results.txt', 'w') as f:
    print(str(all_pred_results), file=f)

# CLASSIFICATION BASED ON N BEST FEATURES
# Since there are big differences in the numbers of audio vs lyrics features, we are looking at different Ns for each group of classifiers
# Max N = no. of features with variance < 0.5
no_of_predictors_aud = [26, 25, 20, 15, 10]
with open('audio_best_predictors.txt', 'w') as f:
    for i in no_of_predictors_aud:
        print(("Results for " + str(i) + " predictors:"),file=f)
        print(str(classify_best_features(audio_features,i)),file=f)

no_of_predictors_lyr = [284, 250, 200, 100]
with open('lyrics_best_predictors.txt', 'w') as f:
    for i in no_of_predictors_lyr:
        print(("Results for " + str(i) + " predictors:"),file=f)
        print(str(classify_best_features(lyrics_features,i)),file=f)

# Now that we know the optimum number of features, let's print the results:
results_audio = classify_best_features(audio_features,26)
plt.bar(results_audio.keys(), results_audio.values())
plt.title('Classification using 26 top audio features')
axes = plt.gca()
axes.set_ylim([0,100]) # Setting y-axis range
for x,y in zip(results_audio.keys(), results_audio.values()): # Adding value labels to the columns
        label = "{:.2f}".format(y)
        plt.annotate(label, (x,y), textcoords="offset points", xytext = (0,2), ha = 'center')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Algorithm', fontsize=12)
plt.savefig('audio_results.png')

results_lyrics = classify_best_features(lyrics_features,250)
plt.bar(results_lyrics.keys(), results_lyrics.values())
plt.title('Classification using 250 top lyrics features')
axes = plt.gca()
axes.set_ylim([0,100])
for x,y in zip(results_lyrics.keys(), results_lyrics.values()):
        label = "{:.2f}".format(y)
        plt.annotate(label, (x,y), textcoords="offset points", xytext = (0,2), ha = 'center')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Algorithm', fontsize=12)
plt.savefig('lyrics_results.png')