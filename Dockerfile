FROM python:3.6-stretch
MAINTAINER Josh 

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for containers
WORKDIR  /usr/src/Temperature_Prediction_ML_Model

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY src/ /src/
RUN ls -la /src/*

# Running Python Application
CMD ["python3", "/Temperature_Prediction.py"]
