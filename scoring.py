import os
import numpy as np
import torch
import cv2
import csv
import pytesseract
import argparse
import warnings
import logging
from time import time
from datetime import datetime
from read_text_pages import read_page
from mmdet.apis import init_detector, inference_detector
from row_inference import column_result_det


warnings.filterwarnings("ignore")

sn_config = "--psm 6 --oem 3 -c tessedit_char_blacklist=1234567890@!#$%^&*()«"
tab_config = "-c tessedit_char_whitelist=()0123456789.LI- --oem 3 --psm 6"


parser = argparse.ArgumentParser(description="Run OCR on a SSR Data Point")

parser.add_argument(
    "--config_file_path",
    "--cfg-fil-pth",
    help="default train config file path",
    default="segmentation/table_det_config.py",
)
parser.add_argument(
    "--checkpoint_file",
    help="checkpoint model path",
    default="segmentation/table_detection.pth",
)
parser.add_argument("--image_path", help="image path")
parser.add_argument("--input_image_directory", help="input image directory")
parser.add_argument("--output_directory", help="output csv directory")

parser.add_argument("--image_year_directory", help="a years image directory")
parser.add_argument(
    "--output_image_year_directory", help="output csv directory for an entire year"
)
args = parser.parse_args()


if args.image_year_directory:
    filename = args.image_year_directory.split("/")[-1] + ".log"
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join("logs", filename),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


logging.info("Loading model into memory")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = init_detector(args.config_file_path, args.checkpoint_file, device="cpu")


def img_inference(image_path: str, output_path: str):
    """ """
    img = cv2.imread(image_path)
    img = image_preprocessing(img)
    # Run Inference
    result = inference_detector(model, img)

    print(str(datetime.now()), ": [INFO] Table Extraction Complete")
    tables_region = [r[:4].astype(int) for r in result[1] if r[4] > 0.85]
    station_names = [r[:4].astype(int) for r in result[0] if r[4] > 0.85]

    if len(tables_region) != 0 and len(station_names) != 0:
        if len(station_names) > 0:
            y_index = {i: y[1] for i, y in enumerate(station_names)}
            y_index = dict(sorted(y_index.items(), key=lambda x: x[1]))
            y_index = {v: k for k, v in y_index.items()}

            new_station_names = []
            for k, v in y_index.items():
                new_station_names.append(station_names[v])

        if len(tables_region) > 0:
            y_index = {i: y[1] for i, y in enumerate(tables_region)}
            y_index = dict(sorted(y_index.items(), key=lambda x: x[1]))
            y_index = {v: k for k, v in y_index.items()}

            new_table_region = []
            for k, v in y_index.items():
                new_table_region.append(tables_region[v])

        def extract_text(config: str, index: int, lang: str, image, gap):
            crop = image[
                index[1] - gap : index[3] + gap, index[0] - gap : index[2] + gap
            ]
            text = pytesseract.image_to_string(crop, config=config, lang=lang)
            return text

        def extract_table_df(config: str, index: int, lang: str, image, gap):
            crop = image[
                index[1] - gap : index[3] + gap, index[0] - gap : index[2] + gap
            ]
            df = column_result_det(crop)
            return df

        sn = [
            extract_text(sn_config, i, lang="eng", image=img, gap=5)
            for i in new_station_names
        ]
        table = [
            extract_table_df(tab_config, i, lang="eng", image=img, gap=10)
            for i in new_table_region
        ]
        print(str(datetime.now()), ": [INFO] Table Columns Complete")

        for i in range(len(sn)):
            w_2csv = [item.split() for item in sn[i].split("\n") if item != ""]

            output_file_name = os.path.join(
                output_path, image_path.split("/")[-1].split(".")[0]
            )
            with open(output_file_name + "_" + str(i) + ".csv", "w") as file:
                writer = csv.writer(file)
                writer.writerows(w_2csv)
            table[i].to_csv(
                output_file_name + "_" + str(i) + ".csv",
                mode="a",
                header=False,
                index=False,
            )
    else:
        output_file_name = os.path.join(
            output_path, image_path.split("/")[-1].split(".")[0] + ".txt"
        )
        read_page(img, output_file_name)
        print(f"[OUTPUT FILE]::> {output_file_name}")


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
    new_result[:, :, 0], new_result[:, :, 1], new_result[:, :, 2] = (
        result,
        result,
        result,
    )

    return new_result


def run_ocr_directory(directory: str, output_dir: str):
    assert os.path.exists(directory), "directory does not exist"
    file_list = [
        os.path.join(directory, ldir)
        for ldir in os.listdir(directory)
        if ldir.endswith(".tif") or ldir.endswith(".tiff")
    ]
    logging.info(f": [INFO] Total number of files in {directory} is {len(file_list)}")
    print(
        str(datetime.now()),
        f": [INFO] Total number of files in {directory} is {len(file_list)}",
    )

    for file in file_list:
        print(f"[FILE] ==> {file}")
        img_inference(file, output_dir)


def preprocess_images(directory: str, output_dir: str):
    assert os.path.exists(directory), "directory does not exist"
    file_list = [
        os.path.join(directory, ldir)
        for ldir in os.listdir(directory)
        if ldir.endswith(".tif") or ldir.endswith(".tiff")
    ]

    for file in file_list:
        print(f"[FILE] ==> {file}")
        img = cv2.imread(file)
        img = image_preprocessing(img)
        cv2.imwrite(
            os.path.join(output_dir, file.split("/")[-1].split(".")[0] + "_3.png"), img
        )


def process_year(input_year_path: str, output_year_path: str):
    """ """
    if not os.path.exists(output_year_path):
        os.mkdir(output_year_path)

    for month in os.listdir(input_year_path):
        if os.path.isdir(os.path.join(input_year_path, month)):
            input_month_path = os.path.join(input_year_path, month)
            # create output directory for each SSR Month if it doesn't exist
            output_month_path = os.path.join(output_year_path, month)
            if not os.path.exists(output_month_path):
                os.mkdir(output_month_path)

            logging.info(f"Start processing month {month.split('-')[-1]}")
            start_time = time()
            print(input_month_path)
            run_ocr_directory(input_month_path, output_month_path)
            print(output_month_path)
            end_time = time()
            seconds_elapsed = end_time - start_time
            logging.info(
                f"Done processing month {month.split('-')[-1]}, Took {seconds_elapsed} seconds"
            )


if __name__ == "__main__":
    if args.input_image_directory and args.output_directory:
        run_ocr_directory(args.input_image_directory, args.output_directory)
    elif args.image_year_directory and args.output_image_year_directory:
        process_year(args.image_year_directory, args.output_image_year_directory)
