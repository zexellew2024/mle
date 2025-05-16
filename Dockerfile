FROM jupyter/pyspark-notebook 
COPY requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /home/jovyan/work 
