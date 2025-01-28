# Data generation code in Python (the only file not done in C++)

import numpy as np

# Parameters for the dataset
num_rows = 100  # Number of rows
num_cols = 100  # Number of columns

# Generate random data
np.random.seed(42)  # For reproducibility
data = np.random.rand(num_rows, num_cols)

# Save the dataset to a file
dataset_file = "Datasetss/d2.txt"
np.savetxt(dataset_file, data, fmt="%.6f", delimiter=" ")

print(f"Dataset saved to {dataset_file}")

#d1 : 10 x 10k
#d2 : 1k x 1k
#d3 : 10k x 10