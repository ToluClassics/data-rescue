import img2pdf
import os


def conv_tiff_pdf(directory, output_pdf, debug=True):
    """
	Function converts tiff files in a directory to a pdf file
	========================================================
	Input:
	directory (string): Directory containing tiff files
	output_pdf (string): File path to output file
	debug(boolean): True if you want to output tiff files
	"""

    assert (os.path.exists(directory) == True), \
        'this directory does not exists'
    files_list = []
    for file in os.listdir(directory):
        if "tif" not in file:
            continue
        files_list.append(directory + "/" + file)

    files_list.sort()

    if debug:
        print(files_list)

    with open(output_pdf, 'a') as f:
        pdf_bytes = img2pdf.convert(files_list)
        f.write(pdf_bytes)


if __name__ == '__main__':
    directory = 'lat_lon_1992'
    output_pdf = 'lat_lon.pdf'
    conv_tiff_pdf(directory, output_pdf)
