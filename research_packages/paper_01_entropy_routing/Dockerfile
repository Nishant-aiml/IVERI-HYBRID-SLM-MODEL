FROM python:3.12-slim
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "proof/verify_everything.py"]
