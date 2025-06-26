# This file is a template, and might need editing before it works on your project.
FROM python:3.11-slim

WORKDIR /usr/src/app

# Install system dependencies if needed (uncomment and add as required)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     postgresql-client mysql-client sqlite3 \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose a port if your app serves HTTP (optional)
# EXPOSE 8000

# Set the default command to run your script (edit as needed)
CMD ["python", "scripts/batchpredict+results.py"]
COPY . /usr/src/app

# For Django
EXPOSE 8000
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]

# For some other command
# CMD ["python3", "app.py"]
