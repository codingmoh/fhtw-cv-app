FROM python:3.8
#separate step is important so we can cache this action
COPY requirements.txt /app/requirements.txt 
WORKDIR /app
RUN pip install -r requirements.txt
COPY . /app
CMD [ "python", "app.py" ]
