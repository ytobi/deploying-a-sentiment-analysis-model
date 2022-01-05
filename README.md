# Sentiment Analysis

Orchestrated a recurrent neural network for the purpose of determining the sentiment of a movie review using the IMDB data set. The model is trained using Amazon's SageMaker. In addition, I deployed the model and construct a simple web app to interact with the deployed model.

[Access the website](http://udl-sentiment-analysis-website.s3-website-us-east-1.amazonaws.com/ "Sentiment analysis endpoint")

# Try it out

[Access the website](http://udl-sentiment-analysis-website.s3-website-us-east-1.amazonaws.com/ "Sentiment analysis endpoint") deployed using AWS S3 bucket. Type your review and submit to examine its sentiment.

# How it works

**model/** : Contains the output of training the model on SageMaker.
**train/** : Contains model definition and code to train pipeline to train the model.
**serve/** : Contains code for deploying an endpoint to AWS Lambda or SageMaker used for inferencing.
**website/** : A ReactJs app to try it out. The app consumes the trained model instance deployed on AWS Lambda.

# See how to train the model

If you are interested in training a yourself, clone this repository to a SageMaker notebook instance.

Run all code cells in the notebook (This will take a very long time to run on a CPU, preferably you should run on a GPU instance eg `ml.m4.xlarger`).

If your interested in creating an end-point to the deployed model, go through the instruction in step 7 of the notebook.
