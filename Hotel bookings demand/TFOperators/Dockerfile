FROM tensorflow/tensorflow:2.4.0
RUN pip install tensorflow_datasets pandas scikit-learn keras
COPY tfjob-hotel-demand.py /
ENTRYPOINT ["python", "/tfjob-hotel-demand.py", "--batch_size", "64", "--learning_rate", "0.1", "--optimizer", "adam"]