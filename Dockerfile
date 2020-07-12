FROM python:3.6-stretch as testml
MAINTAINER Josh 

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for containers
WORKDIR  /home/oem/Desktop/Temperature_Prediction_ML_Model

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY / /
RUN ls -la /*

# Running Python Application
CMD ["python3", "/Temperature_Prediction.py"]
