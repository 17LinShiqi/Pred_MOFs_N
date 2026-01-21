# Pred_MOFs_N
This software is used to quickly predict the adsorption capacity of single/batch MOFs for NH3.

## Usage
This software presents a pre-trained XGB model for predicting adsorption amounts as an interactive desktop application for easy user access. Users do not need to install any other auxiliary software; simply opening Prediction_N_f.exe is sufficient.

## This software has two functions: 
1.**Prediction of adsorption capacity of a single MOF**: After entering the main page, the top section is the single prediction operation interface. Click "READ ME" in the upper left corner to enter the various parameters and obtain the adsorption capacity of a single MOF.
2.**Batch MOF Adsorption Prediction**: After entering the main page, the section at the bottom of the page is the batch prediction operation interface. Users need to follow the instructions in the lower left corner "READ ME" to integrate the parameters of the batch MOFs to be predicted and import them into the software to obtain the adsorption capacity of the batch MOFs.The predicted result will be saved in Result/Batch_Predicted_N.xlsx。

## This folder includes five folders:
1- Code
     1.XGB_code.py that has the code for the machine learning using XGB 
     2.Prediction_N_code.py that has the code for a human-computer interactive interface software.

2- Extrapolation_data
    11.xlsx that is a sample file for batch prediction of material diffusivity.

3- Img 
     full_name.png and sample_file.png that are the interactive interface software required in the illustration picture. 
 
4- model
     xgb.pt that is a trained XGB algorithm model.

5- Result 
     The predicted result will be automatically generated and saved in Result/Batch_Predicted_N.xlsx.
