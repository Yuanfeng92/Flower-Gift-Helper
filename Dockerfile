# Base image
FROM python:3.11-slim-buster

# Updates the system’s package information and then installs the AWS CLI tool in your Docker image
RUN apt update -y && apt install awscli -y

# Set working directory
WORKDIR /app

# Copy the rest of the code
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port Gradio will serve on
EXPOSE 8080

# Command to run the Gradio app
CMD ["python", "app.py"]