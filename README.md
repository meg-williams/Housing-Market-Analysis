
![Static Badge](https://img.shields.io/badge/Linear-Regression-blue?style=for-the-badge&labelColor=%23000080&color=%237FFFD4)  
![Static Badge](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&labelColor=%237FFFD4&color=%23FF3D00)
![Static Badge](https://img.shields.io/badge/tensorflow-keras-orange?style=for-the-badge&labelColor=%23000080&color=%23FF3D00)

# Housing Price Predictions

This project was for a Kaggle Competition. Linear regression was used to predict how much a house might sell for given a select number of features. 


## Dataset

The data used for this project where supplied by Kaggle and included the follow: 

- train.csv: This was the data my model was trained on. It contained housing data that included features of the house and how much it sold for. 
- test.csv: This was the data that my model was tested one. It contained data from different houses but with the same features as the train data and did not include prices. 
- There were a total of 80 features that ranged from overall condition, number of floors, utilities, year built, and more. 
-
## Workflow

1. Preperation: 

    - Any nulls where filled with the mean if the column was numerical and the mode if it was an object. Then any object columns where encoded using a combination of target, ordnal, and one hot enocding. Any columns with only one value, or over 90% of its values where from one value where dropped.  

2. Training: 

    - The training 


## Optimizations




## Lessons Learned

What did you learn while building this project? What challenges did you face and how did you overcome them?


## Appendix

Any additional information goes here