FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /opt

# Setup Dipoorlet
RUN git clone https://github.com/ModelTC/Dipoorlet.git && \
    cd Dipoorlet && \
    python setup.py install
