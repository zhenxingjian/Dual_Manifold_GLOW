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

## Pre-processing
### From dMRI to ODF and DTI
Run the following command. 
``` bash
python process_dMRI.py
```
Note that the mainPATH (line 72) and the name (line 64) need to be modified to suit the correct location and index. Details please refer to the comments in the code.

### Move the data into the same folder with correct file name
Run the following command. 
``` bash
python shuffle_data.py
```
Note that the processed data path (line 9) and the HCP data path (line 20) need to be modified if not store all the files in the default location.

One more thing to explain about why these steps are splitted into several files: the data is very large and fully process from dMRI into DTI and ODF will require longer time. If the processer dumps, you can just re-run the single step file instead of all the pre-processing steps again.

### Compute the eigen values and normalized ODF
Run the following command. 
``` bash
python eig.py
```
Note that the processed data path (line 4) need to be modified if not store all the files in the default location.

The original processed ODF has different sum, which comes from the implementation within [dipy toolbox](https://dipy.org). But by normalize each voxel to have sum of 1, it will be the actual ODF.

One more thing to be mentioned, in our paper, we claim that ODF lies on S^{361} space. This comes from the square root representation of the original ODF. Then, instead of "sum" to be one, it will become "sum of square" to be 1, which is a hyper-sphere.

### Split between train and test
Run the following command. 
``` bash
python Split_train_test.py
```
We also provide the train/ test split that we use in our paper in [train_files.json](train_files.json) and [test_files.json](test_files.json).










