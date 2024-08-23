# FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DIRECTORY=/opt/project
ENV DEBIAN_FRONTEND="noninteractive"
WORKDIR ${DIRECTORY}

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl wget git vim iputils-ping gcc libpq-dev software-properties-common locales  && \
    ln -f -s /usr/bin/python3 /usr/bin/python && \
    echo "es_AR.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    locale -a && \
    export LC_ALL="es_AR.utf8" && \
    export LC_CTYPE="es_AR.utf8" && \
    locale -a

RUN apt install -y libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
    libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libjpeg62


RUN apt-get install -y curl && \
    apt-get install -y python3-dev python3-pip python3-setuptools python3-distutils

#RUN mkdir weights && \
#    cd weights && \
#    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

RUN pip install pipenv
RUN pipenv install --python 3.10.12
RUN pip install git+https://github.com/IDEA-Research/GroundingDINO.git@df5b48a3efbaa64288d8d0ad09b748ac86f22671

# RUN cd ${DIRECTORY} && pip install -e .

CMD tail -f /dev/null
