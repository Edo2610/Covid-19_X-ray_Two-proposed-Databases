# Two Scenarios Covid-19 Recognition on Chest X-Ray Scans using Ensemble CNNs approach

University of Salento and IEMN DOAE Université Polytechnique Hauts-de-France  
Master degree in Computer Engineering  
Supervisor: Cosimo Distante, Abdelmalik Taleb-Ahmed  
Co-supervisor: Fares Bougourzi, Hadid Abdenour 
Student: Edoardo Vantaggiato

___
The recognition of Covid-19 infection from the X-ray images is an emerging field in machine learning and computer vision community. Despite the big efforts that have been made in this field since the appearance of Covid-19 disease (2019), the field still suffers from two drawbacks. First, the available X-ray scans labeled as Covid-19 infected are relatively small. Second, all the works that have been made in the field are separated; no unified data, classes, and evaluation protocol. In this work, based on the public and new collected data, we propose two X-ray covid-19 databases which are: Three-classes Covid-19 ِ and Five-classes Covid-19. For both databases, we test deep learning architectures. In addition, we propose an Ensemble-CNNs approach which outperforms the deep learning architectures and showing promising results in both databases. We make our databases of Covid-19 X-ray scans publicly available to encourage other researchers to use it as a benchmark for their studies.


# Our Source
| | Source | License |
| - | ------ | ------- |
| 1 | [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset) | Apache 2.0, CC BY-NC-SA 4.0, CC BY 4.0 |
| 2 | [Chest X-Ray Images (Pneumonia) from Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) | CC BY 4.0 |
| 3 | [RSNA Pneumonia Detection Challenge from Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) | Open Source |
| 4 | [A Large Chest X-Ray Dataset - CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | Apache 2.0 |
| 5 | [NLM-MontgomerySet](https://lhncbc.nlm.nih.gov/publication/pub9931) | public dataset |
| 6 | [NLM-ChinaCXRSet](https://lhncbc.nlm.nih.gov/publication/pub9931) | public dataset |


# Datasets

### 3-classes dataset

| Class | Train | Val | Test |
| ----- | :---: | :-: | :--: |
| Covid-19 | 404 original + 4848 augmented | 100 | 207 |
| Normal | 404 original + 4848 augmented | 100 | 207 |
| Pneumonia | 404 original + 4848 augmented | 100 | 207 |
| Total | 1212 original + 14544 augmented | 300 | 621 |

:exclamation: for class Covid-19, we use as test set unpublished images collected from Hospitals of Algeria

### 5-classes dataset

| Class | Train | Val | Test | Source |
| ----- | :---: | :-: | :--: | ------ |
| Normal | 404 original + 4848 augmented | 100 | 207 | [2,3,4] |
| Bacetial Penumonia | 404 original + 4848 augmented | 100 | 207 | [2,5,6] |
| Viral Pneumonia | 404 original + 4848 augmented | 100 | 207 | [1,2] |
| Covid-19 | 404 original + 4848 augmented | 100 | 207 | [1] |
| Lung Opacity No Pneumonia | 404 original + 4848 augmented | 100 | 223 | [4] |
| Total | 2020 original + 24240 augmented | 500 | 1051 | - |

:exclamation: for class Covid-19, we use as test set unpublished images collected from Hospitals of Algeria

