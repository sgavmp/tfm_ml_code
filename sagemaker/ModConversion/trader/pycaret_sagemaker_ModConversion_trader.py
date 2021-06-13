import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

import argparse
import datetime
import os

from boto3 import client, resource

# initializing setup
from pycaret.classification import *
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

    _df = pd.read_csv(os.path.join(args.train, 'dataset_ModConversion_traders.csv'))

    clf1 = setup(_df, normalize=True, target='is_converted', categorical_features=['user_country'],
                 fix_imbalance=True, silent=True, html=False, verbose=False)

    top = compare_models(n_select=1, sort='Recall', verbose=False)
    top_tuned = tune_model(top, optimize='Recall', verbose=False)
    calibrated_dt = calibrate_model(top_tuned, verbose=False)
    final_model = finalize_model(calibrated_dt)
    last_metrics = pull()

    save_model(final_model, model_name='ModConversion-trader', verbose=False)
    S3.upload_file('ModConversion-trader.pkl', 'tfm-2021-darwinex', 'models/ModConversion/trader/final_model.pkl')
    S3.upload_file('ModConversion-trader.pkl', 'tfm-2021-darwinex',
                   'models/ModConversion/trader/history/{}_model'.format(datetime.datetime.now()))

    send_metric('ModConversion-trader', 'Accuracy', last_metrics['Accuracy'].iloc[0])
    send_metric('ModConversion-trader', 'AUC', last_metrics['AUC'].iloc[0])
    send_metric('ModConversion-trader', 'Recall', last_metrics['Recall'].iloc[0])
    send_metric('ModConversion-trader', 'Prec.', last_metrics['Prec.'].iloc[0])
    send_metric('ModConversion-trader', 'F1', last_metrics['F1'].iloc[0])
    send_metric('ModConversion-trader', 'Kappa', last_metrics['Kappa'].iloc[0])
    send_metric('ModConversion-trader', 'MCC', last_metrics['MCC'].iloc[0])

    print("saved model!")


def input_fn(input_data, content_type):
    if content_type == "application/json":
        df = pd.DataFrame([pd.read_json(input_data, typ='series')])
        return df
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
    result = predict_model(model, input_data)
    return result


def model_fn(model_dir):
    resource('s3', region_name='eu-west-1').Bucket('tfm-2021-darwinex').download_file(
        'models/ModConversion/trader/final_model.pkl', 'final_model.pkl')
    model = load_model('final_model')
    return model
