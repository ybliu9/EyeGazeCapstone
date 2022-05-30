import os
import json
from time import pthread_getcpuclockid
import parse
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
from script.ocr_preprocess import remove_background
import matplotlib.pyplot as plt

def load_json(path):
	df_raw = pd.read_csv(path, encoding='utf-8')
	format_string = "http://education-annotate.oss-cn-beijing.aliyuncs.com/table_2%2Fpdf2000%2F{}.pdf"
	df_raw['filename'] = df_raw.iloc[:,0].map(lambda x: parse.parse(format_string, x.split("?")[0])[0])
	df_raw['annotated_json'] = df_raw.iloc[:,3]
	return df_raw[['filename', 'annotated_json']]

def generate_ocr_traindata(df, input_directory, output_directory, preprocess=False):

	ocr_table_directory = os.path.join(output_directory, "ocr_traindata/table/") 
	ocr_cell_directory = os.path.join(output_directory, "ocr_traindata/cell/") 

	# Create directory if not exists
	for path in (ocr_table_directory, ocr_cell_directory):
		if not os.path.exists(path):
			os.makedirs(path)

	for i in range(len(df)):
		y = json.loads(df['annotated_json'][i])

		# Find all the non-empty annotation records
		if y:
			filename = df['filename'][i]

			# Parse width, height
			width = y['container']["page1"]["width"]
			height = y['container']["page1"]["height"]
			
			# convert pdf to image
			pdf_path = os.path.join(input_directory, f'{filename}.pdf')
			page = convert_from_path(pdf_path, size=(width*5, height*5))[0]

			try:
				n_cols = sum(d["result"]["类型"]=="表头" for d in y["annotations"])
				n_tables = sum(d["result"]["类型"]=="表格" for d in y["annotations"])
				i_t = 0		
				table_labels = []
				print(f"{filename}:{n_cols} columns, {n_tables} tables")

			except:	# annotation does not contains result if any("result" not in d for d in y["annotations"]):
				print(f"ERROR: {filename} no result: {str(IOError)}")
				continue

			if preprocess:
				open_cv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
				open_cv_image_proc = remove_background(open_cv_image, method="binary", thresh=150, sharpen=True, tozero_thresh = [100,200])
				page = Image.fromarray(open_cv_image_proc[1])

			for i, d in enumerate(y["annotations"]): # loaded json structure in dict y
				x_min = int(min(corner[0] for corner in d["points"]))
				x_max = int(max(corner[0] for corner in d["points"]))
				y_min = int(min(corner[1] for corner in d["points"]))
				y_max = int(max(corner[1] for corner in d["points"]))

				if d["result"]["类型"]=="无效数据": #type==invalid
					print(f"{filename} is invalid")
					break

				elif d["result"]["类型"]=="表格": #type==whole_table
					# save cropped table image
					table_image = page.crop((x_min*5, y_min*5, x_max*5, y_max*5))
					table_file_name = os.path.join(ocr_table_directory, filename)
					table_image.save(f"{table_file_name}_{i_t}.png")


					# save table-wise groundtruth label to txt
					with open(f"{table_label_filename}_{i_t}.gt.txt", "w+") as f:
						f.write("\t".join(table_labels))

					table_labels.clear()
					i_t += 1
					
				else:
					cell_label_filename = os.path.join(ocr_cell_directory, filename)
					table_label_filename = os.path.join(ocr_table_directory, filename)
					label = ''.join(d["label"].splitlines())

					# save cell-wise groundtruth label to txt
					with open(f"{cell_label_filename}_seg_{i}.gt.txt", "w+") as f:
						f.write(label)
						
					table_labels.append(label)

					# save cropped cell image
					cell_image = page.crop((x_min*5, y_min*5, x_max*5, y_max*5))
					cell_image.save(f"{cell_label_filename}_seg_{i}.png")

# /**UNTESTED**/
def generate_mask(df, input_directory, output_directory):
	final_col_directory = os.path.join(output_directory, 'column_mask/') 
	final_table_directory = os.path.join(output_directory, 'table_mask/') 
	final_cell_directory = os.path.join(output_directory, 'cell_mask/') 
	
	# Create directory if not exists
	for path in (final_col_directory, final_table_directory, final_cell_directory):
		if not os.path.exists(path):
			os.makedirs(path)


	for i in range(230, len(df)):
		y = json.loads(df['annotated_json'][i])

		# Find all the non-empty annotation records
		if y:
			filename = df['filename'][i]

			# Parse width, height
			width = y['container']["page1"]["width"]
			height = y['container']["page1"]["height"]
			
			# Create grayscale image array
			col_mask = np.zeros((height, width), dtype=np.int32)
			table_mask = np.zeros((height, width), dtype = np.int32)
			cell_mask = np.zeros((height, width), dtype = np.int32)


			try:
				valid = False
				n_cols = sum(d["result"]["类型"]=="表头" for d in y["annotations"])
				n_tables = sum(d["result"]["类型"]=="表格" for d in y["annotations"])
				i_t = 0		
				print(f"{filename}:{n_cols} columns, {n_tables} tables")

			except:	# annotation does not contains result if any("result" not in d for d in y["annotations"]):
				print(f"\033[1;91mERROR: {filename} no result: {str(IOError)} \033[0m")
				continue
			
			if n_cols==0: 
				# n_cols = count_cols(y["annotations"])
				n_cols = 1
				print(f"\033[1;93mWARNING: {filename} no header \033[0m") 

			col_x_mins, col_x_maxs = [height] * n_cols, [0] * n_cols
			col_y_mins, col_y_maxs = [width] * n_cols, [0] * n_cols


			for i, d in enumerate(y["annotations"]): # loaded json structure in dict y
				x_min = int(min(corner[0] for corner in d["points"]))
				x_max = int(max(corner[0] for corner in d["points"]))
				y_min = int(min(corner[1] for corner in d["points"]))
				y_max = int(max(corner[1] for corner in d["points"]))

				if d["result"]["类型"]=="无效数据": #type==invalid
					print(f"\033[1;91m{filename} is invalid[0m")
					break

				elif d["result"]["类型"]=="表格": #type==whole_table
					# fill column mask
					for ci in range(n_cols):
						col_mask[col_y_mins[ci]:col_y_maxs[ci], col_x_mins[ci]:col_x_maxs[ci]] = 255

					# fill table mask
					table_mask[y_min:y_max, x_min:x_max] = 255

					# clear col_ arrays
					col_x_mins, col_x_maxs = [height] * n_cols, [0] * n_cols
					col_y_mins, col_y_maxs = [width] * n_cols, [0] * n_cols

				else:
					valid = True

					# fill cell mask
					cell_mask[y_min:y_max, x_min:x_max] = 255

					# plt.imshow(page.crop((x_min*5, y_min*5, x_max*5, y_max*5)))
					# plt.title(d["label"])
					# if len(d["label"].splitlines())>1:
					#     continue

					col_x_mins[i%n_cols] = min(col_x_mins[i%n_cols], x_min)
					col_x_maxs[i%n_cols] = max(col_x_maxs[i%n_cols], x_max)
					col_y_mins[i%n_cols] = min(col_y_mins[i%n_cols], y_min)
					col_y_maxs[i%n_cols] = max(col_y_maxs[i%n_cols], y_max)

			if valid:
				im = Image.fromarray(col_mask.astype(np.uint8),'L')
				im.save(final_col_directory + filename + ".jpeg")
				# plt.show(im)

				im = Image.fromarray(table_mask.astype(np.uint8),'L')
				im.save(final_table_directory + filename + ".jpeg")

				im = Image.fromarray(cell_mask.astype(np.uint8),'L')
				im.save(final_cell_directory + filename + ".jpeg")

def show_sample(df, i):

	ant = df["annotated_json"]

	# load JSON
	x = ant[i] #img_0

	# parse x:
	y = json.loads(x)

	n_cols = sum(d["result"]["类型"]=="表头" for d in y["annotations"])
	col_x_mins, col_x_maxs = [float("inf")] * n_cols, [0] * n_cols
	col_y_mins, col_y_maxs = [float("inf")] * n_cols, [0] * n_cols

	return y["annotations"], n_cols   