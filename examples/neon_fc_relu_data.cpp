// Ulrik Hørlyk Hjort
// Fully connected + custom relu example

#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/Tensor.h"

using namespace arm_compute;

int main() {

    // Define tensor shapes
    TensorShape input_shape(10);     
    TensorShape weights_shape(10, 5); 
    TensorShape biases_shape(5);      
    TensorShape output_shape(5);      

    // Create tensors
    Tensor input;
    Tensor weights;
    Tensor biases;
    Tensor fc_output;
    Tensor relu_output;

    input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
    weights.allocator()->init(TensorInfo(weights_shape, 1, DataType::F32));
    biases.allocator()->init(TensorInfo(biases_shape, 1, DataType::F32));
    fc_output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));
    relu_output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

    // Allocate memory for tensors
    input.allocator()->allocate();
    weights.allocator()->allocate();
    biases.allocator()->allocate();
    fc_output.allocator()->allocate();
    relu_output.allocator()->allocate();

    // Load custom data into input, weights, and biases arrays
    float custom_input[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    float custom_weights[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                              1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0};
    float custom_biases[] = {0.1, 0.2, 0.3, 0.4, 0.5};

    std::memcpy(input.buffer(), custom_input, sizeof(custom_input));
    std::memcpy(weights.buffer(), custom_weights, sizeof(custom_weights));
    std::memcpy(biases.buffer(), custom_biases, sizeof(custom_biases));

    // Configure and run the fully connected layer
    NEFullyConnectedLayer fully_connected;
    fully_connected.configure(&input, &weights, &biases, &fc_output);
    fully_connected.run();

    // Apply ReLU activation function
    float *fc_output_data = reinterpret_cast<float *>(fc_output.buffer());
    float *relu_output_data = reinterpret_cast<float *>(relu_output.buffer());
    for (size_t i = 0; i < output_shape.total_size(); ++i)
    {
        // Apply ReLU: Replace negative values with zero
        relu_output_data[i] = std::max(0.0f, fc_output_data[i]);
    }

    // Print the output of the ReLU activation
    std::cout << "Output of ReLU Activation: ";
    for (size_t i = 0; i < output_shape.total_size(); ++i)
    {
        std::cout << relu_output_data[i] << " ";
    }
    std::cout << std::endl;

    // Deallocate memory
    input.allocator()->free();
    weights.allocator()->free();
    biases.allocator()->free();
    fc_output.allocator()->free();
    relu_output.allocator()->free();

    return 0;
}
