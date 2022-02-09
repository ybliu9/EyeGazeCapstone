# Table Extraction via Eye Gaze Tracking
## Overview
In this project, we design a system to automatically extract tables from a document by using eye gaze, and then evaluate this system relative to manual extraction with respect to both speed and accuracy. 

This is a Capstone Project for Columbia University in collaboration with JP Morgan AI Research Institude.

## Setup

### Virtual Environment
For developer and user convienience, we use python virtual environments for dependencies. The virtual environments use python 3.7 and is under the folder "eyegaze".

1. Download the entire *eyegaze* folder in: https://drive.google.com/drive/folders/1u7680FylFu24XL-wcUKhH9chFj1conM-?usp=sharing
into your working directory, and <b>cd to the directory</b>.

2. If you haven't, install venv
  ```
  python install virtualenv
  ```

3. activate and use
  ```
  # activate
  source ./eyegaze/bin/activate
  
  ## Upon successful activation, your command-line should show something like:
  ## (eyegaze) xxx: yyy$                                                   
  ## Now you can start working within the environments
  
  # deactivate
  deactivate
  ```

4. Use Jypyter Notebooks with virtual environments:
  ```
  # install this venv to ipykernel
  python -m ipykernel install --user --name=eyegaze
  ```
Now you are able to choose the virtual environment under 'kernel' tab in your Jupyter Notebook.

### (Alternative) Straight Installation
Alternatively, you can install all dependencies on your python environments:

1. Download the *requirements.txt* file in our project root directory and place in your current working directory. 

2. Install
```
# install all dependencies
pip -r requirements.txt
```
