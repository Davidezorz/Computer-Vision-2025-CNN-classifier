# Computer Vision 2025: CNN Classifier

This repository contains a Convolutional Neural Network (CNN) classifier, developed as part of the Computer Vision 2025 coursework.





## 📦 Prerequisites

Before running the scripts, ensure you have the necessary dependencies installed. 
```bash
pip install -r requirements.txt
```




## 📂 Data Structure

⚠️ Important: The scripts assume your data is organized in a ```.data/``` directory. Ensure your file structure matches the following tree exactly before running the training script:

```
.data/
  ├ train/
  │  ├ Category_1/
  │  │  ├ image_1.jpg
  │  │  ├ ...
  │  ├ Category_2/
  │  │  ├ image___.jpg
  │  │  ├ ...
  │  ├ .../
  │  │  ├ ...
  │  │  ├ ...
  │  └ Category_n/
  │     ├ image___.jpg
  │     └ ...
  └ test/
    ├ ... 
    │  ├ ... 
    │  └ ...
    ├ ... 
    ...
```





## 🚀 Usage

### Run the tests

To run specific unit tests, use the module flag -m. For example:
```
python -m tests.name_of_the_test
```


### Solving Point 1
To solve point one of the exercise, run:
```
python solve.py -config_path='configs/point1.yaml' -train true
```





## Side Comments

─  │
┌ ┐ ╭ ╮
└ ┘ ╰ ╯
┬  ┴  ├  ┤
┼

◀




