# Use Python base image
FROM python:3.10

# Set work directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -U -r requirements.txt

# Run both files in order
# Replace `otherfile.py` with your second file’s name
CMD python train.py & python server.py
