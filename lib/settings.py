import import_env_file
import os

file_path = os.getenv('file_path')
rainfall_column = int(os.getenv('rainfall_column'))
window_size = int(os.getenv('window_size'))
batch_size = int(os.getenv('batch_size'))
shuffle_buffer_size = int(os.getenv('shuffle_buffer_size'))
split_size = int(os.getenv('split_size'))
lr = float(os.getenv('lr'))

