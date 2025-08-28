#Results of the TribiAI_pipeline_demo

This report illustrate the application of different machine learning models on the DLC dataset.
It reproduce the methodology described in the article under reduced subsample of the dataset.
The results differ from the published values, but the logic remain the same

##Models performance

The table presents the models performance based on statistical metrics

| Models          | RMSE    | MAE    | R²    |
|-----------------|---------|--------|-------|
| XGBoost         | 0,0734  | 0,0431 | 0,6625|
| SVM             | 0,0738  | 0,0488 | 0,6586|
| Random Forest   | 0,0714  | 0,0447 | 0,68  |
| KNN             | 0,07    | 0,0436 | 0,693 |
| Extra Trees     | 0,0726  | 0,0418 | 0,6692|
| ANN             | 0,0726  | 0,0509 | 0,6698|

##Figures - Extra Trees example

###Predicted vs actual friction coefficient
![ET Pred vs Actual](Results/Figures/Predicted_vs_Actual_Friction_coefficient/Predicted_vs_actual_CoF_ExtraTrees.png)

###SHAP analysis

![ET SHAP](Results/Figures/SHAP_analysis/SHAP_ExtraTrees.png)

##Figure_models comparison

Models comparison using statistical metrics

###RMSE
![RMSE-Models comparison](Results/Figures/Models_comparison/Models_comparison_RMSE.png)
###MAE
![MAE-Models comparison](Results/Figures/Models_comparison/Models_comparison_MAE.png)
###R²
![R²-Models comparison](Results/Figures/Models_comparison/Models_comparison_R_.png)


