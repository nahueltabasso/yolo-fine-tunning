FROM python:3.9.6
ENV N_PROCESSES=1
ENV DIRECTORY=/opt/project/credit-card-service

RUN mkdir -p ${DIRECTORY}

WORKDIR ${DIRECTORY}

COPY . ${DIRECTORY}
RUN apt-get update && apt-get install -y wget
RUN python -m pip install --upgrade pip
# Install dependencies from requirements
RUN pip install -r requirements.txt
# Install GroundingDINO
RUN pip install git+https://github.com/IDEA-Research/GroundingDINO.git@df5b48a3efbaa64288d8d0ad09b748ac86f22671
RUN mkdir weights && \
    cd weights && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

RUN cd .. && \
    python setup.py

CMD tail -f /dev/null

 