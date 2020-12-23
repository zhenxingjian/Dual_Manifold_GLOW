# Pre-processing steps
## Download the HCP data
- Download the HCP connectome dataset from [CONNECTOMEdb](https://db.humanconnectome.org/). 
You will need to register an accound for downloading.

- Also, as instructed in the webpage, you will need a Plugin called Aspera Connect. Details can be found [Aspera FAQ](http://humanconnectome.org/documentation/connectomeDB/downloading/installing-aspera.html)

- To download the dataset we use for our project, click "[WU-Minn HCP Data - 1200 Subjects](https://db.humanconnectome.org/data/projects/HCP_1200)" in the homepage of HCP. 

- Then click:
  - "Open Group" 
  - "All Family Subjects" 
  - "Download Images" 
  - "Diffusion Preprocessed" 
  - "Download Packages" 
  - "Download Now"
  
  in order

- The Aspera Connect will automaticly pop up. If not, it might because of the delay of the network. Please wait for a while. If still cannot download, please refer to [Aspera FAQ](http://humanconnectome.org/documentation/connectomeDB/downloading/installing-aspera.html) or the instruction in the downloading pages. 

- After downloading, unzip all the zip files. For example, using this bash comments:

  ``` bash
  for i in $(ls *.zip);do unzip $i&&rm $i;done
  ```
  
  This will unzip all the zip files and remove after unzipping.

## Pre-processing, from dMRI to ODF and DTI
``` bash
python process_dMRI.py
```
