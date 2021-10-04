import os
import cv2
import pytesseract


def read_page(input_image, output_file: str, languages=['rus', 'eng']):
    """
    Read text from pages without tables
    ==============================================
    Inputs:
    input_image(string/ numpy array): path to image file or image-array
    output_file(string) : path to output text file
    languages(list) : list of languages to be used by ocr engine
    """
    if type(input_image) == 'str':
        input_image = cv2.imread(input_image)

    languages = '+'.join(languages)
    text = pytesseract.image_to_string(input_image, config="--psm 6 --oem 2",
                                       lang=languages)

    # write to text file
    f = open(output_file, "a")
    f.write(text)
    f.close()


if __name__ == '__main__':
    image = '/Users/mac/Desktop/Page11.tif'
    output_file = 'test.txt'
    read_page(image, output_file)