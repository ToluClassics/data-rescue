import cv2
import torch
import numpy as np
import pandas as pd
import pytesseract
from mmdet.apis import init_detector, inference_detector

config_file_path = 'segmentation/row_det.py'
checkpoint_file = 'segmentation/column_det.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_detector(config_file_path, checkpoint_file, device='cpu')


def table_row_extraction(img, result):
    """

    """
    #print(result[0])
    columns = [r[:4].astype(int) for r in result[0] if r[4] > 0.5]
    columns.sort(key=lambda x: x[0])

    extracted_list_items = []
    for i in range(len(columns)):
        img_crop = img[columns[i][1]-7:columns[i][3]+3, columns[i][0]-1: columns[i][2]+1]
        # random_number = random.randint(0, 999999)
        # cv2.imwrite("/Users/mac/Documents/output/"+str(random_number)+".png", img_crop)
        # cv2.imshow("image", img_crop)
        # cv2.waitKey(0)
        # img_crop = cv2.resize(img_crop, (50, 600))
        x = pytesseract.image_to_string(img_crop, config='-c tessedit_char_whitelist=0123456789()IL- --psm 6 --oem 3', lang='datarescue')
        x = x.replace('L', '1').replace('I', '1')
        extracted_list_items.append(x)

    extracted_list_items = [item.split('\n') for item in extracted_list_items]

    updated_extracted_list_items = []
    for item in extracted_list_items:
        item = [i for i in item if i.strip() != '']
        # print(item)
        updated_extracted_list_items.append(item)

    max_len = max([len(x) for x in updated_extracted_list_items])
    # add zeros to the lists
    temp = [x + [0] * max_len for x in updated_extracted_list_items]
    # Limit the output to the wished length
    updated_extracted_list_items = [x[0:max_len] for x in temp]

    df = pd.DataFrame(list(zip(*updated_extracted_list_items)))
    return df


def column_result_det(img):
    kernel = np.ones((17, 17), np.uint8)
    e = cv2.erode(img, kernel, iterations=2)
    _, th = cv2.threshold(e, 150, 255, cv2.THRESH_BINARY_INV)
    result = inference_detector(model, th)

    df = table_row_extraction(img, result)
    return df


if __name__ == "__main__":
    file = '/Users/mac/Documents/tables_processed/00211_0.png'
    img = cv2.imread(file)
    column_result_det(img)