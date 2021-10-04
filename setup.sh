sudo apt-get update
sudo apt-get install tesseract-ocr

#download custom model from s3
wget https://datarescue-tesseract.s3-us-west-2.amazonaws.com/datarescue.traineddata
wget https://github.com/tesseract-ocr/tessdata/raw/master/eng.traineddata
wget https://github.com/tesseract-ocr/tessdata/raw/master/rus.traineddata



# shellcheck disable=SC2035
mv -v *.traineddata /usr/share/tesseract-ocr/4.00/tessdata/

