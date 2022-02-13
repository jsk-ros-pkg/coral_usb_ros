# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# #==========================================================================

FROM tensorflow/tensorflow:1.12.0-rc2-devel-gpu

# Get the tensorflow models research directory, and move it into tensorflow
# source folder to match recommendation of installation
RUN git clone https://github.com/tensorflow/models.git -b v1.12.0 && \
    mv models /tensorflow/models


# Install the Tensorflow Object Detection API from here
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

# Install object detection api dependencies
RUN apt-get update && \
    apt-get install -y python python-tk
RUN pip install Cython && \
    pip install contextlib2 && \
    pip install pillow && \
    pip install lxml && \
    pip install jupyter && \
    pip install matplotlib

# Get protoc 3.0.0, rather than the old version already in the container
RUN curl -OL "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip" && \
    unzip protoc-3.0.0-linux-x86_64.zip -d proto3 && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-3.0.0-linux-x86_64.zip

# Install pycocoapi
RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    make -j8 && \
    cp -r pycocotools /tensorflow/models/research && \
    cd ../../ && \
    rm -rf cocoapi

# Run protoc on the object detection repo
RUN cd /tensorflow/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

# Set the PYTHONPATH to finish installing the API
ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim

# Install wget (to make life easier below) and editors (to allow people to edit
# the files inside the container)
RUN apt-get update && \
    apt-get install -y wget vim emacs nano

# install gdown
RUN pip install gdown

# install edgetpu compiler 14.1
RUN wget -O libedgetpu1-std_14.1_amd64.deb https://packages.cloud.google.com/apt/pool/libedgetpu1-std_14.1_amd64_c6cb84801d41bb06490d9ee18a0175c2a0b855a5d2865ae76e215a0ca2b9d1a4.deb && \
    dpkg -i libedgetpu1-std_14.1_amd64.deb
RUN wget -O edgetpu-compiler_14.1_amd64.deb https://packages.cloud.google.com/apt/pool/edgetpu-compiler_14.1_amd64_ef6eef29200270dcb941d2c1defa39c7d80e9c6f30cf7ced1c653a30bde0a502.deb && \
    dpkg -i edgetpu-compiler_14.1_amd64.deb

# install tree
RUN apt-get install -y tree

ARG work_dir=/tensorflow/models/research
WORKDIR ${work_dir}

# copy scripts
COPY ./scripts/train.sh /entrypoint.sh
COPY ./scripts ${work_dir}/scripts/

# set ENTRYPOINTS
ENTRYPOINT ["/entrypoint.sh"]
CMD ["dataset"]
