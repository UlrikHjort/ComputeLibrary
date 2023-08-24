import numpy as np

# Generate weights data (10x5 matrix)
weights = np.array([[0.5, 1.0, 1.5, 2.0, 2.5],
                    [3.0, 3.5, 4.0, 4.5, 5.0],
                    [5.5, 6.0, 6.5, 7.0, 7.5],
                    [8.0, 8.5, 9.0, 9.5, 10.0],
                    [10.5, 11.0, 11.5, 12.0, 12.5],
                    [13.0, 13.5, 14.0, 14.5, 15.0],
                    [15.5, 16.0, 16.5, 17.0, 17.5],
                    [18.0, 18.5, 19.0, 19.5, 20.0],
                    [20.5, 21.0, 21.5, 22.0, 22.5],
                    [23.0, 23.5, 24.0, 24.5, 25.0]], dtype=np.float32)

transposed_weights_data = np.transpose(weights)
# Generate biases data (5 elements)
biases = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)

# Save weights and biases to NPY files
np.save("fc_weights.npy", transposed_weights_data)
np.save("fc_biases.npy", biases)

print("NPY files generated successfully.")


