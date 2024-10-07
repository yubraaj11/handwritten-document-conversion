from transformers import VisionEncoderDecoderModel
from transformers import ViTFeatureExtractor, RobertaTokenizer, TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from PIL import Image


import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


import os
import random
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

root_directory = '/content/drive/MyDrive/IIT-HW-Hindi_v1/'

os.listdir(root_directory)


import os
import pandas as pd

# Specify the parent directory path
root_dir = 'HindiSeg/train' # Replace with your directory path

# Initialize an empty list to store file paths
file_paths = []

# Recursively traverse through all directories and subdirectories
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        # Construct the full file path
        file_path = os.path.join(dirpath, filename)
        file_paths.append(file_path)

# Create a DataFrame with the file paths
df = pd.DataFrame(file_paths, columns=['File Path'])

# Display the DataFrame
print(df)

# Print the total number of files
print(f"\nTotal number of files: {len(df)}")

df['File Path'][1]



# Specify the parent directory path
root_dir =  'HindiSeg/test' # Replace with your directory path

# Initialize an empty list to store file paths
file_paths = []

# Recursively traverse through all directories and subdirectories
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        # Construct the full file path
        file_path = os.path.join(dirpath, filename)
        file_paths.append(file_path)

# Create a DataFrame with the file paths
df2 = pd.DataFrame(file_paths, columns=['File Path'])

# Display the DataFrame
print(df2)

# Print the total number of files
print(f"\nTotal number of files: {len(df2)}")

# Specify the parent directory path
root_dir ='HindiSeg/val' # Replace with your directory path

# Initialize an empty list to store file paths
file_paths = []

# Recursively traverse through all directories and subdirectories
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        # Construct the full file path
        file_path = os.path.join(dirpath, filename)
        file_paths.append(file_path)

# Create a DataFrame with the file paths
df3 = pd.DataFrame(file_paths, columns=['File Path'])

# Display the DataFrame
print(df3)

# Print the total number of files
print(f"\nTotal number of files: {len(df3)}")

len(df)+len(df2)+len(df3)

os.chdir(root_dir)

df=df[~df['File Path'].str.endswith('.txt','.py')].reset_index(drop=True)
df2=df2[~df['File Path'].str.endswith('.txt','.py')].reset_index(drop=True)

df3=df3[~df['File Path'].str.endswith('.txt','.py')].reset_index(drop=True)

df.to_csv('train_extracted_files.csv')
df2.to_csv('test_extracted_files.csv')
df3.to_csv('val_extracted_files.csv')

# since the extracted file and the txt file has not all the common files. we perform difference operations

# lets load the files of val,test and train txt to df

with open(root_directory +'val.txt') as f:
    val = f.readlines()
counter = 0
val_list = []
for i in range(len(val)):
    # if counter > 2000:
    #     break
    image_id = val[i].split("\n")[0].split(' ')[0].strip()
#     vocab_id = int(train[i].split(",")[1].strip())
    text = val[i].split("\n")[0].split(' ')[1].strip()
    row = [image_id, text]
    val_list.append(row)
    counter += 1

val_df = pd.DataFrame(val_list, columns=['file_name', 'text'])
val_df.head()

with open(root_directory +'test.txt') as f:
    test = f.readlines()

counter = 0
test_list = []
for i in range(len(test)):
    # if counter > 2000:
    #     break
    image_id = test[i].split("\n")[0].split(' ')[0].strip()
#     vocab_id = int(train[i].split(",")[1].strip())
    text = test[i].split("\n")[0].split(' ')[1].strip()
    row = [image_id, text]
    test_list.append(row)
    counter += 1

test_df = pd.DataFrame(test_list, columns=['file_name', 'text'])
test_df.head()

with open(root_directory+'train.txt') as f:
    train = f.readlines()

counter = 0

train_list = []
for i in range(len(train)):
    # if counter > 5000:
    #     break
    image_id = train[i].split("\n")[0].split(' ')[0].strip()
#     vocab_id = int(train[i].split(",")[1].strip())
    text = train[i].split("\n")[0].split(' ')[1].strip()
    row = [image_id, text]
    train_list.append(row)
    counter += 1

train_df = pd.DataFrame(train_list, columns=['file_name', 'text'])
train_df.head()

# the file paths included iin txt but not present on

def remove_unextracted(path,extract):
  path_set = set(path['file_name'])
  extract_set = set(extract['File Path'])
  return  extract_set - path_set

# def change_full_path(x,root_dir):
#   x['modifed_file_path'] = x['file_name'].apply(lambda x: root_dir + x )

def filter_existing_file_paths(train_df, df):
    """
    Filters 'train_df' to include only rows where the file paths are present in 'df'.

    Parameters:
    train_df (pd.DataFrame): DataFrame containing file paths and corresponding values.
    df (pd.DataFrame): DataFrame containing all existing file paths.

    Returns:
    pd.DataFrame: A new DataFrame with rows from 'train_df' where the file paths are present in 'df'.
    """
    # Extract the file paths from both DataFrames
    train_file_paths = set(train_df['file_name'])
    existing_file_paths = set(df['File Path'])

    # Find the intersection of file paths
    valid_file_paths = train_file_paths.intersection(existing_file_paths)

    # Filter 'train_df' to keep only rows with valid file paths
    filtered_train_df = train_df[train_df['file_name'].isin(valid_file_paths)].reset_index(drop=True)

    return filtered_train_df

valid_train_df = filter_existing_file_paths(train_df,df)
valid_test_df = filter_existing_file_paths(test_df,df2)
valid_val_df = filter_existing_file_paths(val_df,df3)

valid_train_df.to_csv('train_extracted_files.csv')
valid_test_df.to_csv('test_extracted_files.csv')
valid_val_df.to_csv('val_extracted_files.csv')

# thus the finally obtained DF with valid existing files are in terms of
valid_train_df.head(3)

valid_test_df.head(3)

valid_val_df.head(3)

