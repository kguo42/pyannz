#### norm_input

from sklearn.preprocessing import StandardScaler

def return_scaler(input_data):

    # Standardize the data

    scaler_output = StandardScaler()
    if len(input_data.shape) == 1:
        input_data_scaled = scaler_output.fit_transform(input_data.reshape(-1, 1)).flatten()
    else:
        input_data_scaled = scaler_output.fit_transform(input_data)

    return input_data_scaled, scaler_output
  
def transform_val_or_test(input_data, scaler_output):
    #Do this in function to avoid mistake on fit_transform and transform
    
    if len(input_data.shape) == 1:
        input_data_scaled = scaler_output.transform(input_data.reshape(-1, 1)).flatten()
    else:
        input_data_scaled = scaler_output.transform(input_data)
    
    return input_data_scaled