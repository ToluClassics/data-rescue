# import module
from pdf2image import convert_from_path

# Store Pdf with convert_from_path function
images = convert_from_path('/Users/mac/Desktop/SRJune1975.PDF')

print(len(images))
for i in range(len(images)):
    # Save pages as images in the pdf
    images[i].save('/Users/mac/Desktop/training_data/page' + str(i+179) + '.png', 'PNG')