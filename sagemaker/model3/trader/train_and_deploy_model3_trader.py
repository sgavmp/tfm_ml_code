import logging

from boto3 import client
from botocore.exceptions import ClientError
from sagemaker import Session
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer
from sagemaker.session import _wait_until, _deploy_done
from sagemaker.sklearn import SKLearnModel, SKLearn

# sagemaker_session = LocalSession()
# sagemaker_session.config = {'local': {'local_code': True}}
sagemaker_session = Session()

LOGGER = logging.getLogger("sagemaker")
LOGGER.setLevel(logging.INFO)

SM_CLIENT = client('sagemaker')

image = '492253803439.dkr.ecr.eu-west-1.amazonaws.com/pycaret-sagemaker-container'
role = 'arn:aws:iam::492253803439:role/service-role/AmazonSageMaker-ExecutionRole-20210517T174226'
script_path = 'pycaret_sagemaker_model3_trader.py'
instance_type = 'ml.c5.2xlarge'
data_folder = 's3://tfm-2021-darwinex/data/model3/trader/'
job_name = 'model3-trader'
endpoint_name = 'model3-trader-endpoint'

sklearn_preprocessor = SKLearn(
    entry_point=script_path,
    role=role,
    image_uri=image,
    sagemaker_session=sagemaker_session,
    base_job_name=job_name,
    job_name=job_name,
    metric_definitions=[
        {'Name': 'Accuracy', 'Regex': 'Accuracy=(\d\.\d+)'},
        {'Name': 'AUC', 'Regex': 'AUC=(\d\.\d+)'},
        {'Name': 'Recall', 'Regex': 'Recall=(\d\.\d+)'},
        {'Name': 'Prec.', 'Regex': 'Prec.=(\d\.\d+)'},
        {'Name': 'F1', 'Regex': 'F1=(\d\.\d+)'},
        {'Name': 'Kappa', 'Regex': 'Kappa=(\d\.\d+)'},
        {'Name': 'MCC', 'Regex': 'MCC=(\d\.\d+)'}
    ],
    instance_type=instance_type)

sklearn_preprocessor.fit({'train': data_folder})

model = SKLearnModel(image_uri=image, model_data='s3://tfm-2021-darwinex/models/model_dummy.tar.gz',
                     role=role, entry_point=script_path)

try:
    SM_CLIENT.describe_endpoint(EndpointName=endpoint_name)
    SM_CLIENT.delete_endpoint(EndpointName=endpoint_name)
    SM_CLIENT.delete_endpoint_config(EndpointConfigName=endpoint_name)
    desc = _wait_until(lambda: _deploy_done(SM_CLIENT, endpoint_name), 30)
except ClientError as ex:
    pass

predictor = model.deploy(endpoint_name=endpoint_name, initial_instance_count=1,
                         serializer=JSONSerializer(), deserializer=JSONDeserializer(),
                         instance_type=instance_type)

predicted = predictor.predict({
    'user_currency': 'USD',
    'user_country': 40,
    'start_mifid_days': 0.0,
    'has_finished_mifid': 0,
    'finish_mifid_days': None,
    'has_deposit': 0,
    'first_deposit_days': None,
    'first_deposit_amount': 0,
    'first_deposit_platform': 1,
    'mifid_actual_savings': 13,
    'mifid_next_year_savings': 13,
    'mifid_qualifications': 1,
    'mifid_money_other_brokers': 1,
    'mifid_invested_other_brokers': 13,
    'mifid_experience': 1,
    'has_linked_account': 0,
    'linked_account_days': None,
    'has_demo_account': 0,
    'demo_account_days': None,
    'has_demo_trade': 0,
    'demo_trade_days': None,
    'has_mock_account': 0,
    'mock_account_days': None,
    'user_flow_name': 2,
    'days_until_conversion_or_today': 12,
    'is_converted': 0
}
)


print(predicted)

# Not uncomment these lines in production
predictor.delete_model()
try:
    SM_CLIENT.delete_endpoint(EndpointName=endpoint_name)
    _wait_until(lambda: _deploy_done(SM_CLIENT, endpoint_name), 5)
except Exception as ex:
    print(ex)
SM_CLIENT.delete_endpoint_config(EndpointConfigName=endpoint_name)
