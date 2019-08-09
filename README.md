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
<br>
#### 3-2. Implement DeepLab based on `fcn8s_tensorflow.py`
##### 3-2-1. Copy `fcn8s_tensorflow.py` to `deeplab_tensorflow.py`
```
$ cp fcn8s_tensorflow.py deeplab_tensorflow.py
```
##### 3-2-2. Modify `_build_decoder` function in `deeplab_tensorflow.py` so that this function builds deeplab instead of fcn
Note: The only thing you need to take care of is to match the dtype and shape of the final output of this function to the one in `fcn8s_tensorflow.py`
<br>
#### 3-3. Train and evaluate DeepLab on Cityscapes dataset
##### 3-3-1. Copy `fcn8s_tutorial.ipynb` to `deeplab_tutorial.ipynb`
```
$ cp fcn8s_tutorial.ipynb deeplab_tutorial.ipynb
```
##### 3-3-2. Import `deeplab_tensorflow` module instead of `fcn8s_tensorflow`
open `deeplab_tutorial.ipynb` and do the following
```
#from fcn8s_tensorflow import FCN8s # comment out this line
from deeplab_tensorflow import FCN8s # replace fcn8s_tensorflow module with deeplab_tensorflow module
```
##### 3-3-3. Run the notebook and see the result
<br>
#### 3-4. Discussion on FCN and DeepLab
