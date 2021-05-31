import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

import argparse
import datetime
import json
import os

from boto3 import client, resource

# initializing setup
import pandas as pd
from pycaret.arules import *
from sagemaker_containers.beta.framework import (
    encoders, worker)

CLOUDWATCH = client('cloudwatch', region_name='eu-west-1')
S3 = client('s3', region_name='eu-west-1')


def send_metric(job, metric, value):
    CLOUDWATCH.put_metric_data(
        MetricData=[
            {
                'MetricName': metric,
                'Dimensions': [
                    {
                        'Name': 'Model',
                        'Value': job
                    }
                ],
                'Unit': 'None',
                'Value': value
            },
        ],
        Namespace='DarwinexMachineLearningJobs'
    )
    print('Metric: {}={}//'.format(metric, value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    _df = pd.read_csv(os.path.join(args.train, 'dataset_model4.csv'))

    exp_arul101 = setup(data=_df,
                        transaction_id='orderid',
                        item_id='darwin')

    model = create_model(min_support=0.1)
    model.to_pickle('model4.pkl')
    S3.upload_file('model4.pkl', 'tfm-2021-darwinex', 'models/model4/final_model.pkl')
    S3.upload_file('model4.pkl', 'tfm-2021-darwinex', 'models/model4/history/{}_model'.format(datetime.datetime.now()))
    print("saved model!")


def input_fn(input_data, content_type):
    if content_type == "application/json":
        data = json.loads(input_data)
        return data
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    if accept == "application/json":
        return worker.Response(prediction.to_json(), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    result = model[model.antecedents.map(set(input_data.get('darwins')).issubset)]
    return result


def model_fn(model_dir):
    resource('s3', region_name='eu-west-1').Bucket('tfm-2021-darwinex').download_file('models/model4/final_model.pkl', 'final_model.pkl')
    model = pd.read_pickle('final_model.pkl')
    return model
