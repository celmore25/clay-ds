FROM python
ADD . /clay-ds
RUN pip install -r /clay-ds/requirements.txt
