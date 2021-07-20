FROM python:3.8
WORKDIR /keras_model
RUN pip install -U scikit-learn numpy pandas joblib tensorflow
COPY keras.py /keras_model
ENTRYPOINT [ "python", "keras.py" ]
