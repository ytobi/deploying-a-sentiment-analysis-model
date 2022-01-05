from serve.predict import predict_fn, model_fn
import os
import boto3
import random
import sys


def lambda_handler(data, context):

    review = data['review']
    sent_prd = predict_fn(review, model_fn('model/'))

    response = {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": {
            "sentiment": str(sent_prd),
            "review": review,
        },
        "isBase64Encoded": False,
    }


    return response