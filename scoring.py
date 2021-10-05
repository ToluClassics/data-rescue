import os
import os.path as path
import numpy as np
import torch
import cv2
import csv
import pytesseract
import argparse
from read_text_pages import read_page
from mmdet.apis import init_detector, inference_detector


sn_config = "--psm 6 --oem 3 -c tessedit_char_blacklist=1234567890@!#$%^&*()«"
tab_config = "-c tessedit_char_whitelist=0123456789.()LI- --oem 3 --psm 6"


parser = argparse.ArgumentParser(description='Run OCR on a SSR Data Point')

parser.add_argument('--config_file_path', '--cfg-fil-pth', help='default train config file path',
                    default="segmentation/Training_config.py")
parser.add_argument('--checkpoint_file', help='checkpoint model path',
                    default='segmentation/epoch_20.pth')
parser.add_argument('--image_path', help='image  path')
parser.add_argument('--input_image_directory', help='input image directory')
parser.add_argument('--output_directory', help='output csv directory')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_detector(args.config_file_path, args.checkpoint_file, device='cpu')


def img_inference(image_path: str, output_path: str):
    """

    """
    img = cv2.imread(image_path)
    img = image_preprocessing(img)
    # Run Inference
    result = inference_detector(model, img)

    tables_region = [r[:4].astype(int) for r in result[1] if r[4] > .85]
    station_names = [r[:4].astype(int) for r in result[0] if r[4] > .85]

    if len(tables_region) != 0 and len(station_names) != 0:
        if len(station_names) > 1:
            y_index = {i: y[1] for i, y in enumerate(station_names)}
            y_index = dict(sorted(y_index.items(), key=lambda x: x[1]))
            y_index = {v: k for k, v in y_index.items()}

            new_station_names = []
            for k, v in y_index.items():
                new_station_names.append(station_names[v])

        if len(tables_region) > 1:
            y_index = {i: y[1] for i, y in enumerate(tables_region)}
            y_index = dict(sorted(y_index.items(), key=lambda x: x[1]))
            y_index = {v: k for k, v in y_index.items()}

            new_table_region = []
            for k, v in y_index.items():
                new_table_region.append(tables_region[v])

        def extract_text(config: str, index: int, lang: str, image, gap):
            crop = image[index[1] - gap:index[3] + gap, index[0] - gap:index[2] + gap]
            text = pytesseract.image_to_string(crop, config=config,
                                               lang=lang)
            return text

        sn = [extract_text(sn_config, i, lang='datarescue', image=img, gap=5) for i in new_station_names]
        table = [extract_text(tab_config, i, lang='eng', image=img, gap=10) for i in new_table_region]

        for i in range(len(sn)):
            w_2csv = [item.split() for item in sn[i].split('\n') if item != ""]
            f_2csv = [item.split() for item in table[i].split('\n') if item != ""]
            if len(f_2csv[0]) <= 2:
                f_2csv.pop(0)

            f_2csv = [list(filter(('.').__ne__, row)) for row in f_2csv]
            f_2csv = [','.join(row).replace('L', '1').replace('I', '1').split(',') for row in f_2csv]

            output_file_name = os.path.join(output_path, image_path.split("/")[-1].split('.')[0])
            with open(output_file_name + "_" + str(i) + ".csv", "w") as file:
                writer = csv.writer(file)
                writer.writerows(w_2csv)
                writer.writerows(f_2csv)
    else:
        output_file_name = os.path.join(output_path, image_path.split("/")[-1].split('.')[0]+'.txt')
        print(output_file_name)


def image_preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours and remove small noise
    cnts = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 40:
            cv2.drawContours(opening, [c], -1, (0, 0, 0), -1)

    # Invert and apply slight Gaussian blur
    result = 255 - opening
    result = cv2.GaussianBlur(result, (3, 3), 0)

    new_result = np.zeros_like(img)
    new_result[:, :, 0], new_result[:, :, 1], new_result[:, :, 2] = result, result, result

    return new_result


def run_ocr_directory(directory: str, output_dir: str):
    assert(os.path.exists(directory)), "directory does not exist"
    file_list = [os.path.join(directory, ldir) for ldir in os.listdir(directory) if ldir.endswith('.tif') \
                 or ldir.endswith('.tiff')]

    for file in file_list:
        print(f"[FILE] ==> {file}")
        img_inference(file, output_dir)


if __name__ == "__main__":
    run_ocr_directory(args.input_image_directory, args.output_directory)
