###############################################################################
# 1- Import libraries:
###############################################################################   

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import shap
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input

###############################################################################
# 2- Load data:
###############################################################################   
def load_data (file_path,sheet_name):
    #Reading the file
    df=pd.read_excel(file_path, sheet_name) 
    """
    Depending the file format, use different pd.read_ (csv, json...)
    """
    return df

###############################################################################
# 3- Data pretreatment:
###############################################################################  

def prepare_data (df,target_column):
    X=df.drop(columns=[target_column])
    y=df[target_column]
    
    X=X.select_dtypes(include=[np.number]) 
    feature_names=X.columns.tolist() 

    imputer=SimpleImputer(strategy="mean") 
    X_imputed=imputer.fit_transform (X)
    
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform (X_imputed)
    return X,y,scaler,imputer,feature_names,X_imputed,X_scaled

def split_data(X, y, train_size=0.8,test_size=0.2,val_size=None, random_state=42):
    temp_size=1-train_size
    X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=temp_size,random_state=42)
    test_ratio=test_size/(test_size+val_size)
    X_test,X_val,y_test,y_val=train_test_split(X_temp,y_temp,test_size=1-test_ratio,random_state=42)
    return X_train,X_test,y_train,y_test,X_val,y_val

###############################################################################
# 4- Models defintion and training:
###############################################################################  
    
###########################4.1-SKLearn models #################################
def train_models(X_train, y_train):
    models = {

        "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.05, objective='reg:squarederror'),
        "SVM": SVR(kernel='rbf', C=10.0, epsilon=0.02, gamma='scale'),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=25, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=2),
        "Extra Trees": ExtraTreesRegressor(n_estimators=150,max_depth=25, random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

###########################4.2-TensorFlow ANN #################################
def create_nn_model(input_shape):
    model = keras.Sequential([
        Input(shape=(input_shape,)),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
    return model

def train_nn(X_train, y_train, X_val, y_val):
    nn_model = create_nn_model(X_train.shape[1])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, min_delta=1e-4, restore_best_weights=True)
    history = nn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=32, verbose=1, callbacks=[early_stopping])
    return nn_model, history

###############################################################################
# 5-Metrics and plots:
###############################################################################  

################################# Metrics #####################################
def compute_metrics(y_true, y_pred,name,results):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    results[name] = (rmse, r2, mae)
    print(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
    return rmse, r2, mae,results
 
#################### Predicted VS actual y plot ###############################
def plot_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, label=f'{model_name} Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Fit')
    plt.xlabel("Actual Friction Coefficient")
    plt.ylabel("Predicted Friction Coefficient")
    plt.title(f"{model_name}")
    plt.legend()
    plt.show()  
          
####################### Plot shap analysis ####################################
def plot_shap(model, X, model_name,feature_names):
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X,feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary Plot - {model_name}")
        plt.show()
    except Exception as e:
        print(f"SHAP non disponible pour {model_name} : {e}")
        
################ Plot model comparison (metrics comparison) ###################  
def plot_model_comparison(results):
    models = list(results.keys())


    rmse_values = [results[model][0] for model in models]
    r2_values = [results[model][1] for model in models]
    mae_values = [results[model][2] for model in models]

    plt.figure(figsize=(10, 6))
    plt.barh(models, rmse_values, color='skyblue')
    plt.xlabel("RMSE")
    plt.title("Models comparison (RMSE)")
    plt.gca().invert_yaxis()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.barh(models, r2_values, color='lightgreen')
    plt.xlabel("R²")
    plt.title("Models comparison (R²)")
    plt.gca().invert_yaxis()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.barh(models, mae_values, color='salmon')
    plt.xlabel("MAE")
    plt.title("Models comparison (MAE)")
    plt.gca().invert_yaxis()
    plt.show()
        
############################# Plot loss function ##############################
"""
Plot the loss function of the ANN
"""    
def plot_nn_loss(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Artificial Neural Network Training Loss')
    plt.legend()
    plt.show()

###############################################################################
# 7-Running the predictions:
###############################################################################  

def main():
    
    file_path = r"Path to the project\Data\Cherguy & al_DLC_dataset.xlsx"

    sheet_name="Sample_dataset" 
    
    df=load_data (file_path,sheet_name) 

    target_column="Friction coefficient"
    

    X,y,scaler,imputer,feature_names,X_imputed,X_scaled= prepare_data (df,target_column) 
         
    X_train,X_test,y_train,y_test,X_val,y_val = split_data(X_scaled, y, train_size=0.7,test_size=0.2,val_size=0.1, random_state=42)
    
 
    models = train_models(X_train, y_train)
    y_preds={}
    for name, model in models.items():
        y_pred=model.predict(X_test)
        y_preds[name]=y_pred

    nn_model, history = train_nn(X_train, y_train, X_val, y_val)
    y_pred_ann = nn_model.predict(X_test)
    models ["ANN"]=nn_model 
    y_preds ["ANN"] = y_pred_ann 
    
    results={}
    
    for name, model in models.items():
        y_pred=y_preds[name] 
        plot_predictions(y_test, y_pred, name) 
        rmse, r2, mae,results=compute_metrics(y_test, y_pred,name,results) 
        plot_shap(model, X_test, name,feature_names) 
        
    plot_model_comparison(results) 

    return locals()    
vars = main()
globals().update(vars) 

  
# if __name__ == "__main__":
#     main()