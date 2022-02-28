## Classifying music genre with classical machine learning - Special Topics in Digital Humanities 2019/2020 project
This repository the files submitted to the Special Topics in Digital Humanities 2019/2020 (ST) course at Leiden University. 

Author: Kat Kołodziejczyk.

## Data
The data used in this project come from the [Million Song Dataset](http://millionsongdataset.com/). See documentation on the website for a detailed description of the audio features used. Lyrics (in bag-of-words format) come from the [musiXmatch Dataset](http://millionsongdataset.com/musixmatch/).

## Contents
The repository contains the following files:
* [comb_dataset.csv](comb_dataset.csv): csv file with audio features and lyrics per song (50000 most frequent lyrics (rows from "i" to "kad"): frequency of word in given song, 0 if not present)
* [Audio feature names](audio_feature_names.txt): txt file with the name of the 49 audio features used in the classification
* [Classifier](classifier.py): The main Python code
* [STIDH report](#setup): Report I wrote for class

The folder "Results" contains the following output from classifier.py:
* [distribution_genre.png](/Results/distribution_genre.png): Bar plot showing the genre distribution of comb_dataset.csv
* [all_pred_results.txt](/Results/all_pred_results.txt): Accuracy scores (in %) for Gaussian Naive Bayes (GNB), Linear Support Vector Classification (Linear SVC) and k-Nearest Neighbours (KNN) for audio features and lyrics
* [audio_best_predictors.txt](/Results/audio_best_predictors.txt): Accuracy scores for audio features for GNB, Linear SVC, and KNN for 26, 25, 20, 15 and 10 best predictors
* [audio_results.png](/Results/audio_results.png): Bar plot showing the accuracy scores for GNB, Linear SVC, and KNN for 26 best audio predictors
* [lyrics_best_predictors.txt](/Results/lyrics_best_predictors.txt): Accuracy scores for lyrics features for 284, 250, 200 and 100 best predictors
* [lyrics_results.png](/Results/lyrics_results.png): Bar plot showing the accuracy scores for GNB, Linear SVC, and KNN for 250 best lyrics predictors
