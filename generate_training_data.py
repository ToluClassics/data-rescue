import os
import numpy as np
import torch
import cv2
import argparse
from mmdet.apis import init_detector, inference_detector
from scoring import image_preprocessing

parser = argparse.ArgumentParser(description='Run OCR on a SSR Data Point')
parser.add_argument('--config_file_path', '--cfg-fil-pth', help='default train config file path',
                    default="segmentation/table_det_config.py")
parser.add_argument('--checkpoint_file', help='checkpoint model path',
                    default='segmentation/table_detection.pth')
parser.add_argument('--input_image_directory', help='input image directory', required=True)
parser.add_argument('--output_directory', help='output csv directory', required=True)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_detector(args.config_file_path, args.checkpoint_file, device='cpu')


def img_inference(image_path: str):
    """

    """
    img = cv2.imread(image_path)
    img = image_preprocessing(img)
    # Run Inference
    result = inference_detector(model, img)

    tables_region = [r[:4].astype(int) for r in result[1] if r[4] > .85]

    if len(tables_region) != 0:
        if len(tables_region) > 0:
            y_index = {i: y[1] for i, y in enumerate(tables_region)}
            y_index = dict(sorted(y_index.items(), key=lambda x: x[1]))
            y_index = {v: k for k, v in y_index.items()}

            new_table_region = []
            for k, v in y_index.items():
                new_table_region.append(tables_region[v])
        return new_table_region
    return []


def run_ocr_directory(directory: str, output_dir: str):
    assert(os.path.exists(directory)), "directory does not exist"
    file_list = [os.path.join(directory, ldir) for ldir in os.listdir(directory) if ldir.endswith('.tif') \
                 or ldir.endswith('.tiff')]

    for file in file_list:
        print(f"[FILE] ==> {file}")
        image = cv2.imread(file)
        image = image_preprocessing(image)
        img_list = img_inference(file)
        if len(img_list) != 0:
            for i, arr in enumerate(img_list):
                crop = image[arr[1]:arr[3], arr[0]:arr[2]]
                cv2.imwrite(os.path.join(output_dir, file.split('/')[-1].split('.')[0]+'__'+str(i)+'.png'), crop)



"""
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
"""


if __name__ == "__main__":
    run_ocr_directory(args.input_image_directory, args.output_directory)