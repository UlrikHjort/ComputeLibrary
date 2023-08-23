// Ulrik Hærlyk Hjort
// Fully connected example

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
    Tensor output;

    input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
    weights.allocator()->init(TensorInfo(weights_shape, 1, DataType::F32));
    biases.allocator()->init(TensorInfo(biases_shape, 1, DataType::F32));
    output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

    // Allocate memory for tensors
    input.allocator()->allocate();
    weights.allocator()->allocate();
    biases.allocator()->allocate();
    output.allocator()->allocate();

    // Fill input, weights, and biases tensors with random data
    float *input_data = reinterpret_cast<float *>(input.buffer());
    for (size_t i = 0; i < input_shape.total_size(); ++i)
    {
        input_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    float *weights_data = reinterpret_cast<float *>(weights.buffer());
    for (size_t i = 0; i < weights_shape.total_size(); ++i)
    {
        weights_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    float *biases_data = reinterpret_cast<float *>(biases.buffer());
    for (size_t i = 0; i < biases_shape.total_size(); ++i)
    {
        biases_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // Configure and run the fully connected layer with bias
    NEFullyConnectedLayer fully_connected;
    fully_connected.configure(&input, &weights, &biases, &output);
    fully_connected.run();

    // Print the output
    float *output_data = reinterpret_cast<float *>(output.buffer());
    std::cout << "Output: ";
    for (size_t i = 0; i < output_shape.total_size(); ++i)
    {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;

    // Deallocate memory
    input.allocator()->free();
    weights.allocator()->free();
    biases.allocator()->free();
    output.allocator()->free();

    return 0;
}

