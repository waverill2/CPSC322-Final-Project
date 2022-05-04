FROM continuumio/anaconda3:2021.11
ADD . /code
WORKDIR /code
ENTRYPOINT ["python", "all_star_predictions_backend.py"]