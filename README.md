# Post-Proliferation-Prediction

- data_tools.py: a handful of helper functions.
- twitter_scraper.py: contains functions for scraping the text content, author details, and images of Twitter posts.
- data_generator.py: takes data sets and produces a TrainingData object with training and validation partitions. This object can be saved and loaded to ensure models are trained and validated on the same parititons.
- prediction_network.py: produces a model of a specified type (author detail only, text only, or both) using a tokenizer and GLoVE embedding set.
- model_assessment.py: evaluates the accuracy of models on a data set across each classification.