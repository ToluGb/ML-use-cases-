FROM python:3.8
WORKDIR /random forest
RUN pip install -U scikit-learn numpy pandas joblib
COPY rfc.py /random
ENTRYPOINT [ "python", "rfc.py" ]
