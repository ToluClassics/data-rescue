import cv2
import numpy as np
import argparse
import os
import pytesseract
from PIL import Image


def get_num_columns(tess_data, counter):
    """
    Get the number of columns in a table
    ====================================
    inputs:
    tess_data = output of tesseract on the whole image
    counter = index of table in page
    ====================================
    Output:
    length: number of columns
    """
    len_dict = np.zeros(20).tolist()
    max_count = 1

    for j in tess_data[counter - 1]:
        len_dict[len(j.split(","))] += 1

    for j in range(20):
        if max_count < len_dict[j]:
            max_count = len_dict[j]
            length = j
    return length


def tables_from_contours(img, cnt):
    """
    From the contours generated from preprocessing
    Identify the tables and stations regions in the image
    ===============================================
    Inputs:
    img (numpy array): image
    cnt (list): detected contours in image
    ===============================================
    Outputs:
    tables (numpy array): Table arrays
    stations_regions (numpy array): array of station region
    """
    tables = []
    stations_regions = []
    for i in range(len(cnt)):
        area = cv2.contourArea(cnt[i])
        # Extract the regions containing the images
        if area > 100000:
            mask = np.zeros_like(img)
            cv2.drawContours(mask, cnt, i, 255, -1)
            # get rectangle coordinates from the contours
            x, y, w, h = cv2.boundingRect(cnt[i])
            # crop out the table region from the image
            crop = img[y:h + y, x:w + x]
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            masked_img[mask == 0] = 255
            # cut of the upper part of the image containing stations
            crop_left = img[y - 200:y, x:w + x]
            tables.append(crop)
            stations_regions.append(crop_left)
    tables.reverse()
    stations_regions.reverse()
    return tables, stations_regions


def get_station_names(input_image, station_init):
    """
    Returns the name of the stations in each table
    ==============================================
    Inputs:
    input_image(numpy array): image
    station_init(list) : formatted output from tesseract
    """
    config = "--psm 6 --oem 2 eng -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-./"
    img = cv2.imread(input_image, 0)

    h, w = img.shape[:2]
    kernel = np.ones((15, 15), np.uint8)

    e = cv2.erode(img, kernel, iterations=2)
    d = cv2.dilate(e, kernel, iterations=1)
    ret, th = cv2.threshold(d, 150, 255, cv2.THRESH_BINARY_INV)

    mask = np.zeros((h + 2, w + 2), np.uint8)
    # pick out the region containing the table using flood fill
    cv2.floodFill(th, mask, (200, 200), 255)  # position = (200,200)
    # invert colors
    out = cv2.bitwise_not(th)
    out = cv2.dilate(out, kernel, iterations=3)
    # Edge detection
    cnt, h = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    table_stations = []
    stations = ""
    tables, station_regions = tables_from_contours(img, cnt)

    for i, (table, station_region) in enumerate(zip(tables, station_regions)):
        length = get_num_columns(station_init, i)
        length = 10 if length < 8 else length

        width_cutoff = int(table.shape[1] / length)
        # cutoff image regions for different stations
        for j in range(1, length + 1):
            if j != 1:
                start = (j - 1) * width_cutoff - 50
            else:
                start = (j - 1) * width_cutoff
            s = station_region[:, start: j * width_cutoff]

            text = pytesseract.image_to_string(s, config=config)
            text = text.split('\n')[0]
            try:
                stations += text.strip() + ","
            except:
                stations += "N/A,"

        table_stations.append(stations)
        stations = ""

    return table_stations[::-1]
