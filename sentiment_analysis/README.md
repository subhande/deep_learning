# Installation

- Install python 3.8 64 bit
  [Python 3.8](https://www.python.org/ftp/python/3.8.0/python-3.8.0-amd64-webinstall.exe)
- check Add Python 3.8 to PATH
- ![image-20210210134752493](C:\Users\subha\AppData\Roaming\Typora\typora-user-images\image-20210210134752493.png)

```
pip install -r test_requirements.txt
```


# Run command

```
python inference.py
```
# Info

- **Change the inference -> input_file in config.yaml file according to your input**
- **Change the inference -> features_col in config.yaml file according to your input file features column name**

# File Structure

- current folder
    |____ models
            |_____ feedback_best_state_dict.pt
    |_____ inference.py
    |_____ test_requirements.txt


# Dataset Structure

- **feedback**
- **positive**
- **wow**
- **negative**
