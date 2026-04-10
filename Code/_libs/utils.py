import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.utils import shuffle

def load_data(features_path):
  """
  Loads data from CSV files in the specified directory, shuffles it, and maps disease names to integer labels.
  Parameters:
    features_path (str): Path to the directory containing the CSV files.
  Returns:
    all_data (np.ndarray): Array of all data loaded from the CSV files.
    all_target (np.ndarray): Array of mapped target labels.
  """
  diseases = ['MCI', 'MS', 'PD', 'SLA']
  data_path = f'{features_path}'

  list_data = []
  list_target = []
  for disease in diseases:
    disease_path = os.path.join(data_path, disease)
    for file in os.listdir(disease_path):
      with open(os.path.join(disease_path, file), 'rb') as f:
        data = pd.read_csv(f, header=0)
        list_data.append(data)
        list_target.append(disease)

  all_data = np.array(list_data)
  all_target = np.array(list_target)

  all_data, all_target = shuffle(all_data, all_target, random_state=42)

  mapping = {
    "MCI": 0,
    "MS": 1,
    "PD": 2,
    "SLA": 3,
  }
  all_target = np.array([mapping[disease] for disease in all_target])

  print(f"Shape of all data: {all_data.shape}")
  print(f"Shape of target data: {all_target.shape}")
  print(f"Shape of all mapped target data (0, 1, 2, 3): {all_target.shape}")

  return all_data, all_target

def load_psds(folder_path='data/features'):
  """
  Loads PSD features and targets.
  Returns:
    all_data (np.ndarray): Array of PSD features.
    all_target (np.ndarray): Array of mapped target labels.
  """
  features_path = 'PSD'
  diseases = ['MCI', 'MS', 'PD', 'SLA']
  data_path = f'{folder_path}/{features_path}'

  list_data = []
  list_target = []
  for disease in diseases:
    file_path = os.path.join(data_path, f"{disease}.csv")
    data = pd.read_csv(file_path, header=0)
    # Ensure data is a DataFrame and each row is a subject
    list_data.append(data)
    list_target.extend([disease] * len(data))

  # Concatenate all dataframes row-wise (subjects x features)
  all_data_df = pd.concat(list_data, ignore_index=True)
  all_data = all_data_df.values  # shape: (n_subjects, n_features)
  all_target = np.array(list_target)

  # Shuffle data and target together
  all_data, all_target = shuffle(all_data, all_target, random_state=42)

  mapping = {
    "MCI": 0,
    "MS": 1,
    "PD": 2,
    "SLA": 3,
  }
  all_target = np.array([mapping[disease] for disease in all_target])

  print(f"Shape of all data: {all_data.shape}")
  print(f"Shape of target data: {all_target.shape}")
  print(f"Shape of all mapped target data (0, 1, 2, 3): {all_target.shape}")

  return all_data, all_target

def load_atms(folder_path='data/features',zscore=1.6):
  """
  Loads ATM covariance matrices and targets.
  Parameters:
    zscore (float): The z-score threshold to use in the path.
    random_state (int): Random state for reproducibility.
  Returns:
    all_data (np.ndarray): Array of ATM data. 
    all_target (np.ndarray): Array of mapped target labels.
  """
  features_path = f'{folder_path}/ATM/z_tresh_{zscore}/PSD'
  data, target = load_data(features_path)

  return data, target

def load_cov_mats(folder_path='data/features'):
  """
  Loads covariance matrices and targets.
  Returns:
    all_data (np.ndarray): Array of covariance matrices.
    all_target (np.ndarray): Array of mapped target labels.
  """
  features_path = f'{folder_path}/CovarianceMatrices/OAS'
  data, target = load_data(features_path)

  return data, target

def load_corr_mats(folder_path='data/features'):
  """
  Loads Correlation matrices and targeets.
  Returns:
    all_data (np.ndarray): Array of correlation matrices.
    all_target (np.ndarray): Array of mapped target labels.
  """
  feature_path = f'{folder_path}/CC/PSD'
  data, target = load_data(feature_path)

  return data, target

def load_and_combine_csv_files(directory):
    """ 
    This function reads all CSV files in the specified directory and combines them into a single dataframe.
    Expected input files have the following same columns:
      ID,plv_0,plv_1,plv_2,plv_3,plv_4,plv_5,plv_6,...,plv_6669
    The output format will be:
      plv_0,plv_1,plv_2,plv_3,plv_4,plv_5,plv_6,...,plv_6669,ID,class
      
    It has been used to generate the "combined.csv" file in the "v1/data/flatten_datasets/plv" directory. 
    """

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv') and f != 'combined.csv']
    
    # Initialize an empty list to store dataframes
    dataframes = []
    
    # Iterate over the CSV files and read them into dataframes
    for idx, file in enumerate(csv_files):
        df = pd.read_csv(os.path.join(directory, file))
        df.rename(columns={df.columns[0]: 'ID'}, inplace=True)
        # Move the 'ID' column to the end
        id_column = df.pop('ID')
        df['ID'] = id_column
        df['class'] = idx  # Add the 'class' column with the appropriate label
        dataframes.append(df)
    
    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df

def rename_columns_to_edge_id(df):
    """ 
    This function renames the columns of the dataframe to 'edge_id' and returns the modified dataframe.
    Example: Given the input df with columns names:
      plv_0,plv_1,plv_2,plv_3,plv_4,plv_5,plv_6,...,plv_6669,ID,class 
    The function will rename the columns to:
      1-2, 1-3, 1-4, 1-5, 1-6, ..., PLV_115-116,ID,class
    """
    df.columns = [f'PLV_{i}-{j}' for i in range(1, 117) for j in range(i+1, 117)] + ['ID', 'class']
    return df

def rename_class_labels(df):
    """ 
    This function renames the class labels in the dataframe from ['MCI' 'MS' 'PD_off' 'SLA'] to [0 1 2 3].
    """
    df['class'] = df['class'].map({'MCI': 0, 'MS': 1, 'PD_off': 2, 'SLA': 3})
    return df

def find_min_max_range(dir_path):
  """
  This function reads all JSON files in the specified directory and calculates the minimum and maximum balanced accuracy.
  This range is used for the plot of the heatmaps in order to have a consistent color scale.
  """

  min_balanced_accurancy = 1
  max_balanced_accurancy = 0

  for dir in os.listdir(dir_path):
      
    file_path = dir_path + dir + '/log_data.json'
    
    df = pd.read_json(file_path, lines=True)

    df['average_balaced_accuracy'] = df['balanced_accuracies'].apply(lambda x: sum(x) / len(x))

    if df['average_balaced_accuracy'].min() < min_balanced_accurancy:
      min_balanced_accurancy = df['average_balaced_accuracy'].min()

    if df['average_balaced_accuracy'].max() > max_balanced_accurancy:
      max_balanced_accurancy = df['average_balaced_accuracy'].max()
  return min_balanced_accurancy, max_balanced_accurancy
    
def plot_riemeann_results(model, data, target, cv=5):
    """
    This function plots the results of the Riemannian model using a learning curve.
    It takes the model, data, target, and number of cross-validation folds as input.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, data, target, cv=cv, scoring='balanced_accuracy'
    )
    # Calculate the mean and standard deviation of the training and test scores
    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    test_scores_mean = test_scores.mean(axis=1)
    test_scores_std = test_scores.std(axis=1)

    # Plot the learning curve
    plt.plot(train_sizes, train_scores_mean, label='Train')
    plt.plot(train_sizes, test_scores_mean, label='Test')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2)
    plt.title('Learning Curve')
    plt.xlabel('Training Size')
    plt.ylabel('Balanced Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid()
    plt.show()

def upper_triangular_flatten(matrix):
    """
    This function takes a square matrix and returns a flattened array of its upper triangular part.
    It is used to convert the FC matrices (e.g. covariance matrices, ATMs, CC matrices, etc.) into a 1D array for further processing.
    """
    return matrix[np.triu_indices_from(matrix, k=1)]
   

# if __name__ == "__main__":
    
#     directory = 'v1/data/flatten_datasets/plv'

#     combined_df = load_and_combine_csv_files(directory)
#     renamed_df = rename_columns_to_edge_id(combined_df)

#     # Save the combined dataframe to a CSV file
#     renamed_df.to_csv(f'{directory}/combined.csv', index=False)
