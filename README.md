# Piazza Label Predictor
Created an program that automatically identify the subject of posts from Piazza 
<br>using natural language processing and machine learning techniques. 

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

## What I Learned
- Linked Lists
- Binary Search Trees
- Maps<>
- Templates
- Recursion
- Handling CSV files
- Multi-Variate Bernoulli Naive Bayes Classifier
