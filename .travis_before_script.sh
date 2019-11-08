echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std
sudo apt install python3-edgetpu
sudo apt install python3-pip
if [ "$ROS_DISTRO" == "kinetic" ]; then
  wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp35-cp35m-linux_x86_64.whl
  pip3 install tflite_runtime-1.14.0-cp35-cp35m-linux_x86_64.whl
  rm tflite_runtime-1.14.0-cp35-cp35m-linux_x86_64.whl
fi
if [ "$ROS_DISTRO" == "melodic" ]; then
  wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp36-cp36m-linux_x86_64.whl
  pip3 install tflite_runtime-1.14.0-cp36-cp36m-linux_x86_64.whl
  rm tflite_runtime-1.14.0-cp36-cp36m-linux_x86_64.whl
fi
