FROM pytorch/pytorch:latest
WORKDIR /project
RUN apt update
########## opencv ############
RUN apt install -y libsm6 libxext6 libxrender-dev libglib2.0-0 cmake
RUN pip install future scipy
RUN pip install opencv-python
RUN pip install opencv-contrib-python
RUN pip install jupyter
RUN pip install matplotlib
RUN pip install torchgeometry sklearn scipy future
RUP pip install scikit-learn
ENV SHELL /bin/bash
