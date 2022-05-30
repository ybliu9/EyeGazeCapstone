# Table Extraction via Eye Gaze Tracking
## Overview

This is a Capstone Project for the Data Science Institute of Columbia University in collaboration with JP Morgan AI Research Team.

In this project, we designed a system incorporating Computer Vision (CV) and Optical Character Recognition (OCR) techniques to automatically detect tables from a document, determine the table of interest by tracking eye gaze, and then extract texts within the tables. 

The eye gaze tracking technology used was Gazepoint's eye tracker GP3, and its device control software **Gazepoint Control** and **Gazepoint Analysis** were available under Windows OS. 

Major toolboxes used: [pyGaze](https://github.com/esdalmaijer/PyGaze), [openCV](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html), [pyTorch](https://pytorch.org/get-started/locally/#windows-installation), [pytesseract](https://github.com/madmaze/pytesseract), [YOLOv5](https://github.com/ultralytics/yolov5)

Key code has been wrapped up in an independent package named [**pyGazeTE**](https://github.com/ybliu9/pygazeTE). 

Please find the full tutorial at https://medium.com/@yl4616/table-extraction-and-text-recognition-from-images-via-eye-gaze-tracking-ae22a707263 

## Setup

### Step 1. Clone the project GitHub repo. 
You would need to download this repo to a local directory or clone it via Git Bash/GitHub Desktop.

```
git clone https://github.com/ybliu9/EyeGazeCapstone.git
```
### Step 2. Unzip pyGazeTE module and move it under ./site-packages. 
All code for the pipeline has been wrapped in a python module called **pygazeTE**, and you can also find it in the repo at **EyeGazeCapstone/lib/pygazete.zip**. Unzip the file and move it to the path where your other python packages are saved, then you can import it as a normal package and use it outside this project in the future.

### Step 3. Create a virtual environment and install dependencies. 
For developer and user convenience, we use Python virtual environments in **Anaconda3** for dependencies. The virtual environments use python 3.8.

(1) Open **Anaconda Prompt** and create a virtual environment named **eyegaze** and activate the venv:
```
conda -V     #check if conda is installed
## create venv with your python version
conda create -n eyegaze python=3.8 anaconda  
conda activate eyegaze    #activate virtual environment
```
(2) Cd to the EyeGazeCapstone directory:
```
## For Windows OS:
cd ./GitHub/EyeGazeCapstone/Table_extraction    #change working directory
## Upon successful activation, your command-line should show something like:
## (eyegaze) xxxx>
## Now you can start working within the environments
## to deactivate
conda deactivate
```
(3) Install dependencies required
```
pip install -r requirements.txt
```
(4) Use Jypyter Notebooks with virtual environments:
```
## install this venv to ipython kernel
ipython kernel install --user --name=pygaze
```
Now you are able to choose the virtual environment under the "kernel" tab in your Jupyter Notebook.

### Step 4. Connect to Gazepoint¡¯s eye tracking device

If you are using an eye tracker from Gazepoint, you can quickly set up your device following this [instruction manual](https://www.gazept.com/dl/gazepoint_quick_start.pdf).

Please download the device control software **Gazepoint Control** from the official website https://www.gazept.com/downloads/ (Downloads are protected with a password, which can be found in the manual).

Install and open **Gazepoint Control** and find the **IP address** of your device under Gazepoint Settings, just like the screenshot shown below. Your address can be a different one.

![](https://miro.medium.com/max/724/0*nsekb-6IrzOPlUPn)

### Step 5. Run the table extraction Jupyter Notebook

Open **pygazeTE-tutorial.ipynb** in Jupyter Notebook, configure several important variables following the guide, and run the whole notebook. Make sure you use your virtual environment as the kernel.
