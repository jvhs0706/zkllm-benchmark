from datasets import load_dataset
import os

if not os.path.exists('data'):
    os.makedirs('data')

# c4
traindata = load_dataset(
    'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
)
traindata.save_to_disk('data/c4-train')

valdata = load_dataset(
    'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
)
valdata.save_to_disk('data/c4-val')

# c4_new
traindata = load_dataset(
    'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
)
traindata.save_to_disk('data/c4_new-train')
valdata = load_dataset(
    'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
)
valdata.save_to_disk('data/c4_new-val')