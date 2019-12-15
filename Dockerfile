FROM python:3.7

RUN pip install -U pip

RUN pip install numpy tensorflow==1.15.0 opencv-python Cython

WORKDIR /tmp
RUN git clone https://github.com/ildoonet/tf-pose-estimation

WORKDIR /tmp/tf-pose-estimation

RUN pip install -r requirements.txt

COPY diff.patch .
RUN patch -p1 < diff.patch && \
    python setup.py install

WORKDIR /bone
RUN git clone https://github.com/teamnameis/ml

COPY kimono.pickle .
COPY *.py ./

CMD [ "python", "server.py" ]
