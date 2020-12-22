ARG ROS_DISTRO
ARG UBUNTU_VERSION
FROM osrf/ros:${ROS_DISTRO}-desktop-${UBUNTU_VERSION}
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y wget curl git
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN echo "deb http://packages.ros.org/ros-testing/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
# RUN wget http://packages.ros.org/ros.key -O - | apt-key add -

# FIXME: https://github.com/start-jsk/jsk_apc/pull/2664
RUN apt-get update && apt-get dist-upgrade -y && apt-get install -y \
    apt-utils \
    python-catkin-tools \
    python-rosdep \
    python-setuptools \
    python-wstool \
    python-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install 'pip==9.0.3' && \
    pip install 'setuptools==44.1.0'

RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates && \
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y libedgetpu1-legacy-std python3-edgetpu python3-pip && \
    rm -rf /var/lib/apt/lists/*

ARG ROS_DISTRO
RUN apt-get update && \
    if [ ${ROS_DISTRO} = "kinetic" ]; then \
      wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp35-cp35m-linux_x86_64.whl && \
      pip3 install tflite_runtime-1.14.0-cp35-cp35m-linux_x86_64.whl; \
      pip3 install --user opencv-python==4.2.0.32 numpy\<1.19.0; \
    else \
      wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp36-cp36m-linux_x86_64.whl && \
      pip3 install tflite_runtime-1.14.0-cp36-cp36m-linux_x86_64.whl; \
      apt-get install python3-opencv; \
    fi && \
    rm -rf /var/lib/apt/lists/*

ARG ROS_DISTRO
RUN cd ~ && \
    mkdir -p ros/${ROS_DISTRO}/src && \
    cd ros/${ROS_DISTRO}/src && \
    wstool init && \
    wstool set coral_usb_ros https://github.com/knorth55/coral_usb_ros.git -v master --git -y && \
    wstool up -j 2 && \
    wstool merge -y coral_usb_ros/fc.rosinstall && \
    wstool merge -y coral_usb_ros/fc.rosinstall.${ROS_DISTRO} &&\
    wstool up -j 2

# /opt/ros/${ROS_DISTRO}/share can be changed after rosdep install, so we run it 3 times.
RUN rosdep update --include-eol-distros && \
    apt-get update && \
    for i in $(seq 3); do \
      rosdep install --rosdistro ${ROS_DISTRO} -r -y -i --from-paths /opt/ros/${ROS_DISTRO}/share ~/ros/${ROS_DISTRO}/src; \
    done && \
    rm -rf /var/lib/apt/lists/*

ARG ROS_DISTRO
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    cd ~/ros/${ROS_DISTRO} && \
    if [ ${ROS_DISTRO} = "kinetic" ] ; then \
      catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so; \
    else \
      catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so ; \
    fi && \
    catkin build
