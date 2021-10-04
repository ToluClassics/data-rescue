import csv

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
    s = string.strip()[:3].lower()
    try:
        out = m[s]
        return out
    except:
        raise ValueError('Not a month')


flag = 0
year = 1964

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]
months_31 = ["January", "March", "May", "July", "August", "October", "December"]

consolidated_data = []
consolidated_length = 0
for year in range(1980, 1992):
    data = []
    print(year)
    if year >= 1998 and year <= 2003:
        for i in range(12):
            data.append([])
            if i + 1 == 2:
                if year % 4 == 0:
                    for j in range(29):
                        data[i].append('\0')
                else:
                    for j in range(28):
                        data[i].append('\0')
            elif i + 1 == 1 or i + 1 == 3 or i + 1 == 5 or i + 1 == 7 or i + 1 == 8 or i + 1 == 10 or i + 1 == 12:
                for j in range(31):
                    data[i].append('\0')
            else:
                for j in range(30):
                    data[i].append('\0')

    else:
        months_tmp = []
        filename = "Moscow/" + str(year) + ".csv"
        for i in range(12):
            data.append([])
        with open(filename, 'rb') as f:
            content = f.read().splitlines()
            i = 0
            while (i < len(content)):
                try:
                    line = str(content[i], 'utf_8').strip().split(',')
                except:
                    line = str(content[i]).strip().split(',')
                # print(line[0])
                if line[0] == "Day":
                    print("reached")
                    flag = 1
                    counter = 0
                    for j in range(1, len(line)):
                        if line[j] == 'F':
                            continue
                        else:
                            months_tmp.append(line[j])
                            if line[j] == months[counter]:
                                counter += 1
                                continue
                            else:
                                while line[j] != months[counter]:
                                    # data.append([])
                                    print(line[j], months[counter])
                                    if counter + 1 == 2:
                                        if year % 4 == 0:
                                            for p in range(29):
                                                data[counter].append('\0')
                                        else:
                                            for p in range(28):
                                                data[counter].append('\0')
                                    elif counter + 1 == 1 or counter + 1 == 3 or counter + 1 == 5 or counter + 1 == 7 or counter + 1 == 8 or counter + 1 == 10 or counter + 1 == 12:
                                        for p in range(31):
                                            data[counter].append('\0')
                                    else:
                                        for p in range(30):
                                            data[counter].append('\0')
                                    counter += 1
                                counter += 1
                    last_month = line[len(line) - 1]
                    j = 2
                    while last_month == "F":
                        last_month = line[len(line) - j]
                        j -= 1
                    # print(last_month)
                    last_month_ind = month_string_to_number(last_month)
                    for p in range(last_month_ind, 12):
                        if p + 1 == 2:
                            if year % 4 == 0:
                                for j in range(29):
                                    data[p].append('\0')
                            else:
                                for j in range(28):
                                    data[p].append('\0')
                        elif p + 1 == 1 or p + 1 == 3 or p + 1 == 5 or p + 1 == 7 or p + 1 == 8 or p + 1 == 10 or p + 1 == 12:
                            for j in range(31):
                                data[p].append('\0')
                        else:
                            for j in range(30):
                                data[p].append('\0')

                else:
                    if flag == 1:
                        j = 1
                        if line[0] != "SUM" and line[0] != "MEAN":
                            while j < len(line) - 1:
                                # print(j, int(j/2), months_tmp, len(months_tmp), line[j])
                                if months_tmp[int(j / 2)] == "February":
                                    if line[0] == "30" or line[0] == "31":
                                        j += 2
                                        continue
                                    if line[0] == "29":
                                        if year % 4 == 0:
                                            data[month_string_to_number(months_tmp[int(j / 2)]) - 1].append(line[j])
                                    else:
                                        data[month_string_to_number(months_tmp[int(j / 2)]) - 1].append(line[j])
                                elif line[0] == "31":
                                    if months_tmp[int(j / 2)] in months_31:
                                        # print(line[j])
                                        data[month_string_to_number(months_tmp[int(j / 2)]) - 1].append(line[j])
                                    else:
                                        j += 2
                                        continue
                                else:
                                    # print(line[j], len(data))
                                    data[month_string_to_number(months_tmp[int(j / 2)]) - 1].append(line[j])
                                j += 2
                i += 1

    with open("Moscow/converted_data/" + str(year) + ".csv", mode='w') as output:
        writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        total_len = 0
        for i in range(len(data)):
            for j in range(len(data[i])):
                row = [str(year) + "-" + str(i + 1) + "-" + str(j + 1), data[i][j]]
                consolidated_data.append([str(year) + "-" + str(i + 1) + "-" + str(j + 1) + " " + data[i][j]])
                writer.writerow(row)
                total_len += 1
        print(year, total_len)
        consolidated_length += total_len

with open("Moscow/converted_data/consolidated_data.txt", mode='w') as output:
    writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in consolidated_data:
        writer.writerow(row)
print("Consolidated_length", consolidated_length)
