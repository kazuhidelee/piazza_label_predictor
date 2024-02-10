# Piazza Label Predictor
<img width="948" alt="Screenshot 2024-02-10 at 12 33 47 AM" src="https://github.com/kazuhidelee/piazza_label_predictor/assets/122251831/5782e4f2-d03c-4b73-aa5e-6e844a87a90c">

Created an program that automatically identify the subject of posts from Piazza and which label/folder/tag the post belong to
using natural language processing and machine learning techniques. 
<br>The program will read in CSV files as training data and output the predicted labels of the posts in the testing files based on the training data

## Bag of Words Model
Treat a Piazza post as a bag of words” - each post is simply characterized by which words it includes. 
<br>The ordering of words is ignored, as are multiple occurrences of the same word. 
<br>These two posts would be considered equivalent:
- "When is the midterm exam"
- "exam exam is when when"

## Prediction
<img width="837" alt="Screenshot 2024-02-05 at 11 42 39 PM" src="https://github.com/kazuhidelee/piazza_label_predictor/assets/122251831/0d2fbac5-fc73-4e7e-b3a6-787a8f721617">

## Usage
```$ make main.exe ```

<br>```$ ./main.exe TRAIN_FILE TEST_FILE ```

<br>```$ ./main.exe TRAIN_FILE TEST_FILE [--debug]``` (enabling the debug flag give more detailed calculated statistics)
<br> The predicted lables will be based on the training file provided in the command line

## What I Learned
- Linked Lists
- Binary Search Trees
- Maps
- Templates
- Recursion
- Handling CSV files
- Multi-Variate Bernoulli Naive Bayes Classifier
