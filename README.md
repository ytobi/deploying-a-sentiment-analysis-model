# Deploying a Sentiment Analysis Model with Amazon SageMaker.

In this project, I construct a recurrent neural network for the purpose of determining the sentiment of a movie review using the IMDB data set. The model is constructed using Amazon's SageMaker service. In addition, I deployed the model and construct a simple web app to interact with the deployed model.


The notebook and Python files provided here, once completed, result in a simple web app which interacts with a deployed recurrent neural network performing sentiment analysis on movie reviews. This project assumes some familiarity with SageMaker.

# Installation

Clone this repository to a SageMaker notebook instance.

You need not modify anything, you only have to include your API url at the specified location inside the webpage. Simple webpage is found in `website/index.html`

# Usage

### Run
Run all code cells in the notebook (This will take a very long time to run on a CPU, preferably you should run on a GPU instance eg `ml.m4.xlarger`).

If your interested in creating an end-point to the deployed model, go through the instruction in step 7 of the notebook.


