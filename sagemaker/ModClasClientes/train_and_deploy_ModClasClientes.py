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
script_path = 'pycaret_sagemaker_ModClasClientes.py'
instance_type = 'ml.c5.2xlarge'
data_folder = 's3://tfm-2021-darwinex/data/ModClasClientes/'
job_name = 'ModClasClientes'
endpoint_name = 'ModClasClientes-endpoint'

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

predicted = predictor.predict({'userid': '1ad1801b-d20c-4397-8378-07ac4e248441',
                               'has_darwin': True,
                               'pfees': 0.0,
                               'darwinia': 0.0,
                               'dscore': 0.0,
                               'commissions': 3.066004793590487,
                               'investment': 0.0,
                               'old_darwin': False})
print(predicted)

# Comment these lines in production
predictor.delete_model()
try:
    SM_CLIENT.delete_endpoint(EndpointName=endpoint_name)
    _wait_until(lambda: _deploy_done(SM_CLIENT, endpoint_name), 5)
except Exception as ex:
    print(ex)
SM_CLIENT.delete_endpoint_config(EndpointConfigName=endpoint_name)
