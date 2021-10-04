<h1>Data Rescue Project</h1>

This is the repo for converting pdf tabular data into a machine readable (csv) format. Below are the different modules:

<ul>
<li><b>ocr.py:</b> The main file for converting numbers and station names of a pdf page into csv file.</li>
<br>
<li><b>get_stations.py:</b> Used in ocr.py to convert station names into machine readable text using image segmentation and tesseract.</li>
<br>
<li><b>convert_table_to_csv.py:</b> A wrapper file to convert pdf files of all the years available to csv. It iterates over all the files and for each one calls the ocr.py to convert that page's data into csv format. It is used by search.py file to search the data for a particular station.</li>
<br>
<li><b>search.py:</b> This file is used to get data for all the years available for a particular station and save the data into csv file for different years.</li>
<br>
<li><b>reformat_data.py:</b> This file converts the data generated from search.py into the required format i.e. (location, id, date, data).</li>
<br>
<li><b>convert_tiff_to_pdf.py:</b> Convert multiple tiff images to a single pdf file.
</ul>
<br>

# To Replicate this project

In this section, we describe the steps required to setup and retrain the categorization model.

- Clone the project

```
$ git clone https://github.com/gwf-uwaterloo/data-rescue.git
```

- Create a virtual environment, install requirements
```
$ python3 -m venv env

(env) $ source env/bin/activate
(env) $ pip install -r requirements.txt

More to follow....

```