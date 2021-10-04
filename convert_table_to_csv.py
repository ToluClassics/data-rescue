import os

directory = "TIFF/"
temp_dir = directory
for year in range(1980, 1992):
	directory = os.path.join(temp_dir, str(year))
	for month in os.listdir(directory):
		if month == ".DS_Store":
			continue
		print(month, year)
		new_path = os.path.join(directory,month)
		new_path = os.path.join(new_path, "data")

		for _ in os.walk(new_path):
			files = _

		for filenames in files[2]:
			if "tif" in filenames:
				print(month,filenames)
				file_path = os.path.join(new_path, filenames)
				command = " str(python ocr.py -i " + file_path + " -f " + month)
				os.system(command)
