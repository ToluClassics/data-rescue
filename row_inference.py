import cv2
import numpy as np
import pandas as pd
import pytesseract

result = np.load('/Users/mac/Desktop/file.npy')
file = '/Users/mac/Desktop/xy.png'

img = cv2.imread(file)


columns = [r[:4].astype(int) for r in result if r[4] > 0]
columns.sort(key = lambda x: x[0])

lii = []
for i in range(len(columns)):
    img_crop = img[columns[i][1]:columns[i][3], columns[i][0]: columns[i][2]]
    #img_crop = cv2.resize(img_crop, (50, 600))
    x = pytesseract.image_to_string(img_crop, config='-c tessedit_char_whitelist=0123456789()IL- --psm 6 --oem 3')
    lii.append(x)

lii = [item.split('\n') for item in lii]

lii2 = []
for item in lii:
    item = [i for i in item if i.strip() != '']
    #print(item)
    lii2.append(item)


max_len = max([len(x) for x in lii2])
# add zeros to the lists
temp = [x + [0]*max_len for x in lii2]
# Limit the output to the wished length
lii2 = [x[0:max_len] for x in temp]

df = pd.DataFrame(list(zip(*lii2)))

print(df)



#cv2.imshow('image', img_crop)
#cv2.waitKey(0)