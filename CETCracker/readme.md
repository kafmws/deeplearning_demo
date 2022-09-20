# CETCracker

**For learning purpose only.**

A simple CNN to recognize captcha on the [website](https://passport.neea.edu.cn/CETLogin?ReturnUrl=https://cet-bm.neea.edu.cn/Home/VerifyPassport/?LoginType=0)

> This is my homework on the class *Image processing*.

## directory structure

```file
CETCracker
    │  augmentor.py
    │  binarization.py
    │  datasets.py
    │  models.py
    │  predict.py
    │  readme.md
    │  shuffle.py
    │  train.py
    │  
    ├─checkpoints
    │      model_cnnv2_0.75.pth
    │      model_cnnv2_0.79.pth
    │      
    └─rawdata
        ├─test  
        │       test.jpg
        │      
        └─train 
                train.jpg
```

dir  `rawdata` contains two subdir `train` and `test`, which contains 625 and 100 captcha images respectively collect from the website as our original dataset. The dataset has been divided by `shuffle.py`.
file `augmentor.py` uses `augmentor` to generate more data samples for `data augmentation`.
`binarization.py` coverts `RGB` images to grayscale images.
`models.py` defines a simple CNN network.

dir `checkpoints` contains two weight files, which reach `0.75` and `0.79` recognition accuracy on the 100 test images respectively.

---

## start
```bash
git clone https://github.com/kafmws/deeplearning_demo.git
cd CETCracker

pip install pytorch augmentor opencv-python-headless
python predict.py
```

---

## results

- distribution of characters in dataset

<img src="https://cdn.jsdelivr.net/gh/kafmws/pictures/notes/distribution of characters in dataset.png" alt="distribution of characters in dataset" width="80%">

&emsp;&emsp;

- model architecture

<img src="https://cdn.jsdelivr.net/gh/kafmws/pictures/notes/model Architecture.png" alt="model Architecture" width="80%">

&emsp;&emsp;

- Experimental setup and the results

<img src="https://cdn.jsdelivr.net/gh/kafmws/pictures/notes/Experimental setup and results of CNN.png" alt="Experimental setup and results of CNN" width="80%">

## Reference

https://github.com/ice-tong/pytorch-captcha