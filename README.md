## FCN-8s implementation in TensorFlow
[original repo](https://github.com/pierluigiferrari/fcn8s_tensorflow)


### 1. Install Packages
```
$ sudo apt-get -y install libsm6 libxext6 libxrender-dev
$ pip install moviepy matplotlib opencv-python jupyter numpy scipy==1.0
```

### 2. Download Data
```
$ bash download.sh
```

### 3. Project Description
#### 3-1. Train and evaluate FCN on Cityscapes dataset
see `fcn8s_tutorial.ipynb`
#### 3-2. Implement DeepLab based on `fcn8s_tensorflow.py`
##### 3-2-1. copy `fcn8s_tensorflow.py` to `deeplab_tensorflow.py`
```
cp fcn8s_tensorflow.py deeplab_tensorflow.py
```
##### 3-2-2. modify `_build_decoder` function in `deeplab_tensorflow.py` so that this function builds deeplab instead of fcn
Note: The only thing you need to take care of is to match the dtype and shape of the final output of this function to the one in `fcn8s_tensorflow.py`
