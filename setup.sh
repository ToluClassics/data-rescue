sudo apt-get update
sudo apt-get install tesseract-ocr

#download custom model from s3
wget https://datarescue-tesseract.s3-us-west-2.amazonaws.com/datarescue.traineddata
wget !wget https://github.com/tesseract-ocr/tessdata/blob/master/eng.traineddata
wget https://github.com/tesseract-ocr/tessdata/raw/master/rus.traineddata



# shellcheck disable=SC2035
mv -v *.traineddata /usr/share/tesseract-ocr/4.00/tessdata/

#x = pytesseract.image_to_string(img2, config='-c tessedit_char_whitelist=-0123456789()IL --psm 6 --oem 3')