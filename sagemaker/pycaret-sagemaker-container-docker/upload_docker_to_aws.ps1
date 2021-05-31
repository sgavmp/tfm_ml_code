(Get-ECRLoginCommand).Password | docker login --username AWS --password-stdin 492253803439.dkr.ecr.eu-west-1.amazonaws.com
docker build -t pycaret-sagemaker-container .
docker tag pycaret-sagemaker-container:latest 492253803439.dkr.ecr.eu-west-1.amazonaws.com/pycaret-sagemaker-container:latest
docker push 492253803439.dkr.ecr.eu-west-1.amazonaws.com/pycaret-sagemaker-container:latest