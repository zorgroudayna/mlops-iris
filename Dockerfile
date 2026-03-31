# base image
FROM python:3.10-slim

# set working directory
WORKDIR /app

# copy requirements first (for caching)
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy all project files
COPY . .

# expose port
EXPOSE 8000

# run the api
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]