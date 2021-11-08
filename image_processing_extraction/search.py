import os
from PIL import Image
# import pytesseract
import argparse
import cv2
import csv
from difflib import get_close_matches
from collections import OrderedDict

words_list = ["Moscow", "NormanWells"]


def month_string_to_number(string):
    m = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12
    }
    print(string[0])
    s = string[0][:3].lower()
    try:
        out = m[s]
        return out
    except:
        raise ValueError('Not a month')


# words_list = ["Stockholm"]
stations = {}
numerals = []
for i in range(10):
    numerals.append(str(i))

alphabets = []
for i in range(26):
    alphabets.append(chr(i + 65))
    alphabets.append(chr(i + 97))

for files in os.listdir("results"):
    if files == ".DS_Store" or files == "res":
        continue
    print(files)
    month = files.split(".")[0].split("_")[0]
    # print(month)
    year = files.split(".")[0].split("_")[1]
    with open("results/" + files) as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        station_names = []
        pg_no = 0
        for row in file_reader:
            flag = 0
            count = 1
            # print(row)
            if len(row[0]) == 0:
                continue
            if row[0][0] not in alphabets and row[0][0] not in numerals:
                continue
            counter = 0
            # print(row)
            for data in row:
                counter += 1
                if "COUNTRY" in data or "Sum" == data or "Mean" == data:
                    flag = 2
                    break
                if "Page" in data:
                    pg_no = data.split('Page')[1]
                    break
                if "STATIONS" in data:
                    flag = 1
                    # print(month,data)
                    station_names = []
                # station_names.append(data)
                if flag == 1:
                    if len(data) == 0:
                        continue
                    if len(get_close_matches(data, words_list)) != 0:
                        # print(data)
                        dt = get_close_matches(data, words_list, cutoff=0.8)
                        if len(dt) > 0:
                            data = dt[0]
                        # print(data,"2")
                    # if(len(data) > 0):
                    # print(data,"2")
                    # data = data[0]
                    # else:
                    #	break
                    # print(get_close_matches(data, words_list))
                    if len(data) > 0:
                        station_names.append(data)
                    # print(stations.keys())
                    if data in stations.keys():
                        city_dict = stations[data]
                    else:
                        stations[data] = {}
                    city_dict = stations[data]
                    if year in city_dict:
                        year_dict = city_dict[year]
                    else:
                        year_dict = {}
                    if month in year_dict:
                        month_dict = year_dict[month]
                    else:
                        year_dict[month] = {}
                    month_dict = year_dict[month]
                    month_dict["data"] = []
                    month_dict["pg_no"] = pg_no
                    year_dict[month] = month_dict
                    city_dict[year] = year_dict
                    stations[data] = city_dict
                else:
                    # print(row, files)
                    # print(files)
                    # print(station_names, counter-1, count, row)
                    # print(row)
                    # print(files)
                    # print(station_names[counter-1], counter-1, len(station_names), data)
                    city_dict = stations[station_names[counter - 1]]
                    # if station_names[counter-1] == "MoscowUniv.":
                    #	print(station_names, city_dict, stations["Moscow"])
                    year_dict = city_dict[year]
                    month_dict = year_dict[month]
                    month_dict["data"].append(row[count - 1])
                    year_dict[month] = month_dict
                    city_dict[year] = year_dict
                    stations[data] = city_dict
                    count += 1

                if counter == 10:
                    break

result_dict = stations.get("Moscow")
print(result_dict['1989']['December'])

sorted_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"]
for year in result_dict.keys():
    sub_dict = result_dict[year]
    if year == "1989":
        print(sub_dict["December"])
    # sorted_sub_dict = OrderedDict(sorted(sub_dict.items(), key=month_string_to_number))
    cols = []
    cols.append("Day")
    for key in result_dict.keys():
        cols.append(key)
        cols.append('F')
    with open("Moscow/" + str(year) + ".csv", 'w') as out:
        ocr_writer = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ocr_writer.writerow(cols)
        pg_no = ""
        for month in sub_dict.keys():
            sub_sub_dict = sub_dict[month]
            # print(sub_sub_dict)
            pg_no += "Page " + str(sub_sub_dict['pg_no']) + ","
        # ocr_writer.writerow(pg_no.split(','))
        for i in range(31):
            row = str(i + 1) + ","

            for month in sorted_months:
                # print(i, month)
                if month not in sub_dict.keys():
                    continue
                if month == "December" and year == "1989":
                    print(sub_dict[month]['data'][i])
                try:
                    row += str(sub_dict[month]['data'][i]) + ","
                    row += '0,'
                except:
                    continue
            ocr_writer.writerow(row.split(','))
