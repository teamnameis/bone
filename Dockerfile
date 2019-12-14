FROM python:3.7

RUN pip install -U pip

RUN pip install numpy tensorflow==1.15.0 opencv-python Cython

RUN git clone https://github.com/ildoonet/tf-pose-estimation && \
    cd tf-pose-estimation && \
    pip install -r requirements.txt && \
    python setup.py install

RUN pip install flask

COPY server.py .

CMD [ "python", "server.py" ]
