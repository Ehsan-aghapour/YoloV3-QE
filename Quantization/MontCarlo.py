import numpy as np

# Generate a random binary sequence of length 75
binary_sequence = np.random.randint(2, size=75)

# Find the indices where the value is 1
indices_of_ones = np.where(binary_sequence == 1)[0]

print("Random Binary Sequence:", binary_sequence)
print("Indices of Positions with Value 1:", indices_of_ones)
