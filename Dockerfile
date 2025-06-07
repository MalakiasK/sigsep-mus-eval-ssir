FROM continuumio/miniconda3

WORKDIR /museval-ssir

RUN apt-get update && apt-get install -y curl
RUN apt-get -y install libsndfile1-dev
RUN pip install -U pip wheel
RUN conda install numpy scipy cffi
RUN conda install -c conda-forge ffmpeg==3.4 libsndfile
RUN git clone https://github.com/MalakiasK/sigsep-mus-eval-ssir /src && pip install -e /src

ENTRYPOINT ["/opt/conda/bin/museval-ssir"]
