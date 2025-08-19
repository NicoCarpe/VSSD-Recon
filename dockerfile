# Specify the base image
FROM pytorch/pytorch:2.0.1-cuda12.2-cudnn8-devel

# Copy the example directory to /app
COPY test /app

# install dependencies from requirements.txt
RUN pip install -r /app/configs/requirements.txt

# DO NOT EDIT THE FOLLOWING LINES
COPY *_run.py /
COPY submitter.json /
# You can run more commands when the container start by 
# editing docker-entrypoint.sh
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/bin/bash", "/docker-entrypoint.sh"]