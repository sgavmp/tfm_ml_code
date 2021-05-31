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
from pycaret.regression import *
from sagemaker_containers.beta.framework import (
    encoders, worker)

CLOUDWATCH = client('cloudwatch', region_name='eu-west-1')
S3 = client('s3', region_name='eu-west-1')
WINDOW = 27


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

    _df = pd.read_csv(os.path.join(args.train, 'dataset_model2.csv'))

    for i in range(1, WINDOW + 1):
        col = 'lag_' + str(i)
        _df[col] = _df['incomes'].shift(i)

    _df.dropna(inplace=True)

    exp_reg101 = setup(data=_df, normalize=True, ignore_features=['date'], data_split_shuffle=False,
                       target='incomes', remove_perfect_collinearity=False, silent=True, html=False,
                       verbose=False)
    best = compare_models(exclude=['ransac'], sort='RMSLE')
    best_tuned = tune_model(best)
    final_model = finalize_model(best_tuned)
    last_metrics = pull()

    save_model(final_model, model_name='model2', verbose=False)
    S3.upload_file('model2.pkl', 'tfm-2021-darwinex', 'models/model2/final_model.pkl')
    S3.upload_file('model2.pkl', 'tfm-2021-darwinex', 'models/model2/history/{}_model'.format(datetime.datetime.now()))

    send_metric('model2', 'MAE', last_metrics['MAE'].iloc[0])
    send_metric('model2', 'MSE', last_metrics['MSE'].iloc[0])
    send_metric('model2', 'RMSE', last_metrics['RMSE'].iloc[0])
    send_metric('model2', 'R2', last_metrics['R2'].iloc[0])
    send_metric('model2', 'RMSLE', last_metrics['RMSLE'].iloc[0])
    send_metric('model2', 'MAPE', last_metrics['MAPE'].iloc[0])

    print("saved model!")


def input_fn(input_data, content_type):
    if content_type == "application/json":
        df = pd.read_json(json.loads(input_data))
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
    df = input_data.dropna()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(df.date)

    df['prediction'] = df['incomes']
    df['calculated'] = 0

    columns = ['lag_' + str(i) for i in range(1, WINDOW + 1)]
    columns.reverse()

    dates_to_predict = input_data[input_data.incomes.isnull()].date

    for row in dates_to_predict:
        if row.weekday() != 5:
            df_temp = df[df.index < row].tail(n=WINDOW).transpose().iloc[[2]]
            df_temp.columns = columns
            prediction = predict_model(model, df_temp).iloc[0]['Label']
            new_row = pd.DataFrame({'prediction': prediction, 'calculated': 1}, index=[row])
            if not df[df.index.isin([row])].empty:
                df.update(new_row)
            else:
                df = df.append(new_row)
            df.sort_index(inplace=True)

    return df


def model_fn(model_dir):
    resource('s3', region_name='eu-west-1').Bucket('tfm-2021-darwinex').download_file('models/model2/final_model.pkl',
                                                                                      'final_model.pkl')
    model = load_model('final_model')
    return model
