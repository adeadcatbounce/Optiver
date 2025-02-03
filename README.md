This is the code for solving, Optiver : Trading at the Close (Kaggle) competition problem. I use gradient boost (LightGBM) to solve this problem. The code ingests the training dataset, develops features, fits a LightGBM regressor to the dataset and makes predictions on the dataset. The target variable is, (Future 60 seconds WAP_Stock - Future 60 seconds WAP_SyntheticIndex). The feature importance plot is shown below. Then the code evaluates the quality of predictions by using statistical metrics such as MSE, RMSE, MAE and R_Squared. Finally, the code makes predictions on the test dataset

![image alt](https://github.com/adeadcatbounce/Optiver/blob/83b62d3e45df83745d6b41723f4268b0a22d4dea/feature_importance.png)

