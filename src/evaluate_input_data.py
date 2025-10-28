#### Evaluate input data

from scipy.stats import pearsonr
import numpy as np

def test_correlation(input_data_x, input_data_y):
    print('')
    if len(input_data_x.shape)>1:
        for i in range(input_data_x.shape[1]):
            if len(input_data_y.shape)>1:
                for j in range(input_data_y.shape[1]):
                    corr, p_value = pearsonr(input_data_x[:, i], input_data_y[:, j])
                    print(f"Feature {i}: correlation = {corr:.4f}, p-value = {p_value:.4e}")
            else:
                corr, p_value = pearsonr(input_data_x[:, i], input_data_y)
                print(f"Feature {i}: correlation = {corr:.4f}, p-value = {p_value:.4e}")
    else:
        corr, p_value = pearsonr(input_data_x, input_data_y)
        print(f"Feature {i}: correlation = {corr:.4f}, p-value = {p_value:.4e}")

    print('')


def show_stats(input_data_x, input_data_y):

    if np.isnan((input_data_x)).any() or np.isnan((input_data_y)).any():
        print('⚠️', 'NaNs in your input, please check')

    if len(input_data_x.shape)>1:
        for i in range(input_data_x.shape[1]):
            print(f"Feature {i}: mean: {np.mean(input_data_x[:, i])}, min,max: {np.min(input_data_x[:, i]), np.max(input_data_x[:, i])}, std: {np.std(input_data_x[:, i])}")
    else:
        print(f"input X: mean: {np.mean(input_data_x)}, min,max: {np.min(input_data_x), np.max(input_data_x)}, std: {np.std(input_data_x)}")
    
    if len(input_data_y.shape)>1:
        for i in range(input_data_y.shape[1]):
            print(f"input Y: mean: {np.mean(input_data_y[:, i])}, min,max: {np.min(input_data_y[:, i]), np.max(input_data_y[:, i])}, std: {np.std(input_data_y[:, i])}")
    else:
        print(f"input Y: mean: {np.mean(input_data_y)}, min,max: {np.min(input_data_y), np.max(input_data_y)}, std: {np.std(input_data_y)}")
    

    if np.max(np.abs(input_data_x)) >5: 
        print('⚠️', f'Input X range may lead to saturation for activation functions, Normalization recommended.')

    if np.max(np.abs(input_data_y)) >5: 
        print('⚠️', f'Input Y range may lead to saturation for activation functions, Normalization recommended.')

    
