# Table Extraction via Eye Gaze Tracking
## Overview

This is a Capstone Project for the Data Science Institute of Columbia University in collaboration with JP Morgan AI Research Team.

In this project, we designed a system incorporating Computer Vision (CV) and Optical Character Recognition (OCR) techniques to automatically detect tables from a document, determine the table of interest by tracking eye gaze, and then extract texts within the tables. 

The eye gaze tracking technology used was Gazepoint's eye tracker GP3, and its device control software *Gazepoint Control* and *Gazepoing Analysis* were available under Windows OS. 

Major toolboxes used: pygaze, openCV, pyTorch, pytesseract

Key code has been wrapped up in an independent package named *pyGazeTE*. 

## Setup

### Clone Github Repo
You would need to download this repo to a local directory or clone this repo in via Git Bash/Github Desktop.
```
  git clone https://github.com/ybliu9/EyeGazeCapstone.git
  cd EyeGazeCapstone      #working directory
  ```

### Virtual Environment
For developer and user convienience, we use python virtual environments in Anaconda3 for dependencies. The virtual environments use python 3.8 and is under the folder "eyegaze".

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
pip install -r requirements.txt
```
