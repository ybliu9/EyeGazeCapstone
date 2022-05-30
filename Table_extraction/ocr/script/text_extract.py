import cv2
import pytesseract as pt
from itertools import repeat
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageFilter
from script.ocr_preprocess import get_structure, get_borders

def extract(test_file_path, output_text_path, method="table"):
	"""
    reads image of tabular data and perform ocr recognition 
    and returns recognized text result with table structure
    
    arguments
    -----------
	test_file_path		string, path for input file to be recognized
	output_text_path	string, the output path for txt file
    method          	string, ("table", "cell"), the ocr recognition mode
    
    returns			
    -----------
    data         	string, structured text of ocr recognizition result
    
    """
	assert method in ("table", "cell")

	image = cv2.imread(test_file_path)
	raw = image.copy()
	
	# adjust image
	image = adjust(image)
	_, image = remove_background(image, method="binary", thresh=150, sharpen=True, tozero_thresh = [100,200])

	# show preview
	show_images(raw, image)

	# OCR 
	# config options: https://stackoverflow.com/questions/44619077/pytesseract-ocr-multiple-config-options

	if method=="table":
		## Table-wise mode (--psm=6)
		data = pt.image_to_string(image, lang='fintabnet_full', config='--psm 6') 
		print(data)

	elif method=="cell":
		# get the table structure
		thresh_values = get_structure(image, binary=True)

		# coordinates for border lines
		listx, listy = get_borders(thresh_values, plot=False, kernel1 = 7, erode0_iter=1, erode1_iter=1, 
							dilate0_iter=2, dilate1_iter=3,
							stripe1=6,
							img=img, img_name=img_name)

		# record coordinates pairs
		coords = list()
		for i in range(len(listx)-1):
			coords.append((listx[i],listx[i+1],listy[0],listy[-1])) #(x_min, x_max, y_min, y_max)

		# ocr prediction
		rows = list()
		for i, coord in enumerate(coords):
			print(coord)
			x_min, x_max, y_min, y_max = coord
			cell_image = image[x_min:x_max, y_min:y_max]
			row_data = pt.image_to_string(cell_image, lang='fintabnet_full', config='--psm 7') 
			# print(row_data)
			rows.append(row_data)

		data = "".join(rows)


	# write data to txt
	with open(output_text_path, "w+") as f:
		f.write(data)

	return data