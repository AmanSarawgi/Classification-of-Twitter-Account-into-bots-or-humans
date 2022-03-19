# Classification-of-Twitter-Account-into-bots-or-humans

## Introduction

This project was developed as part of the ***Final Year Major Project*** I had enrolled in during my **Bachelors in Computer Engineering** at ***University of Mumbai***.

## Problem Statement

Goal of the project was to built a model based on the twitter accounts details which could classify the accounts into bots and humans. This was a Supervised Classification problem and we trained the model using machine learning algorithms like Decision Tree, Random Forest and K-Nearest Neighbor.

## Dataset and Research Methods

Dataset included 1.6 Million records of tweets from different users. Just like any other Natural Language Processing task, a lot of effort went into preprocessing the data (tweets). Two main approaches viz. Stemming and Lemmatization were explored to generate tokens and a document term matrix was used to train a Support Vector Machine, in order to predict the users sentiment (positive or negative).


Detailed analysis, approach, implementation and results can be found in the [Python File](./BotDetection.py)

Also find the [Project Report](./Advance_Machine_Learning_Final_Report.pdf)

## Observation Results and Conclusion

Support Vector Machine was found to be performing decently really well on the dataset with a accuracy of over 75-77%. Performance of the Support Vector Machine developed here, matched and was even slightly outperformed by the Naive Bayes classifier built using same preoprocessing steps.

Hence Naive Bayes Classifier built on Lemmatization as a preprocessing step is recommended for the task of ***Sentiment Analysis of Tweets***.
