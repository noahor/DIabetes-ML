FROM python:3.8.5 as base
WORKDIR /app
COPY requirements.txt /app
COPY DataToPredict.csv /app
COPY Diabetes_dataset.csv /app
COPY DIabetes_Exercise_Full_Sloution.py /app
RUN pip install -r ./requirements.txt
CMD ["python", "DIabetes_Exercise_Full_Sloution.py"]~

#### Debug ####

