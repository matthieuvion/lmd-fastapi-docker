# Use the pre-configured FastAPI image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Set the working directory within the container
WORKDIR /app

# Copy the requirements file to the container
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the application files to the container
COPY ./app /app/app

# Make sure the model directory exists (it will contain the downloaded model)
RUN mkdir -p /app/app/model

# Copy the download_model script to the container
COPY ./download_model.py /app/app/download_model.py

# Run the download_model.py script to download the model and tokenizer from HuggingFace into the container
RUN python /app/app/download_model.py

# Expose the port the app runs on
EXPOSE 5100

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5100"]
