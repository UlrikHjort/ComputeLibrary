#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include <iostream>

using namespace arm_compute;

int fc_data()
{
    // Define tensor shapes
    TensorShape input_shape(10);       
    TensorShape weights_shape(10, 5);  
    TensorShape biases_shape(5);       
    TensorShape output_shape(5);       

    // Create tensors
    Tensor input;
    Tensor fc_weights;
    Tensor fc_biases;
    Tensor fc_output;

    // Initialize tensors
    input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
    fc_weights.allocator()->init(TensorInfo(weights_shape, 1, DataType::F32));
    fc_biases.allocator()->init(TensorInfo(biases_shape, 1, DataType::F32));
    fc_output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

    // Allocate memory for tensors
    input.allocator()->allocate();
    fc_weights.allocator()->allocate();
    fc_biases.allocator()->allocate();
    fc_output.allocator()->allocate();    

    
    // Define and copy weights, biases, and input data
    float weights_data[] = {
        0.5, 1.0, 1.5, 2.0, 2.5,
        3.0, 3.5, 4.0, 4.5, 5.0,
        5.5, 6.0, 6.5, 7.0, 7.5,
        8.0, 8.5, 9.0, 9.5, 10.0,
        10.5, 11.0, 11.5, 12.0, 12.5,
        13.0, 13.5, 14.0, 14.5, 15.0,
        15.5, 16.0, 16.5, 17.0, 17.5,
        18.0, 18.5, 19.0, 19.5, 20.0,
        20.5, 21.0, 21.5, 22.0, 22.5,
        23.0, 23.5, 24.0, 24.5, 25.0
    };
    float biases_data[] = {0.5, 1.0, 1.5, 2.0, 2.5};
    float input_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    std::memcpy(fc_weights.buffer(), weights_data, sizeof(weights_data));
    std::memcpy(fc_biases.buffer(), biases_data, sizeof(biases_data));
    std::memcpy(input.buffer(), input_data, sizeof(input_data));

    // Configure and run the fully connected layer
    NEFullyConnectedLayer fully_connected;
    fully_connected.configure(&input, &fc_weights, &fc_biases, &fc_output);
    fully_connected.run();

    // Print the output of the fully connected layer
    float *fc_output_data = reinterpret_cast<float *>(fc_output.buffer());
    for (size_t i = 0; i < output_shape[0]; ++i) {
        std::cout << "Output[" << i << "] = " << fc_output_data[i] << std::endl;
    }

    return 0;
}






#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/core/Utils.h"
#include "utils/Utils.h"

using namespace arm_compute;

int fc_npy_data(){


    // Define tensor shapes
    TensorShape input_shape(10);       
    TensorShape weights_shape(10, 5);  
    TensorShape biases_shape(5);       
    TensorShape output_shape(5);       

    // Create tensors
    Tensor input;
    Tensor fc_weights;
    Tensor fc_biases;
    Tensor fc_output;

    // Initialize tensors
    input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
    fc_weights.allocator()->init(TensorInfo(weights_shape, 1, DataType::F32));
    fc_biases.allocator()->init(TensorInfo(biases_shape, 1, DataType::F32));
    fc_output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

    // Allocate memory for tensors
    input.allocator()->allocate();
    fc_weights.allocator()->allocate();
    fc_biases.allocator()->allocate();
    fc_output.allocator()->allocate();

    try {
    arm_compute::utils::NPYLoader npy_loader2;            
    // Load biases from NPY file
    npy_loader2.open("./examples/fc_data/fc_biases.npy");
    //npy_loader2.init_tensor(fc_biases, DataType::F32);
    npy_loader2.fill_tensor(fc_biases);

    } catch (const std::exception& e) {
    std::cerr << "Error2 loading NPY files: " << e.what() << std::endl;
    return 1;  
    }
    
    try {
    // Load data from NPY files using NPYLoader
    arm_compute::utils::NPYLoader npy_loader1;



    // Load weights from NPY file
    npy_loader1.open("./examples/fc_data/fc_weights.npy");

    //npy_loader1.init_tensor(fc_weights, DataType::F32);    
    npy_loader1.fill_tensor(fc_weights);
    } catch (const std::exception& e) {
    std::cerr << "Error1 loading NPY files: " << e.what() << std::endl;
    return 1;  
    }



    
    // Generate input data (example values)
    float input_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::memcpy(input.buffer(), input_data, sizeof(input_data));

    // Configure and run the fully connected layer
    NEFullyConnectedLayer fully_connected;
    fully_connected.configure(&input, &fc_weights, &fc_biases, &fc_output);
    fully_connected.run();

    // Print the output of the fully connected layer
    float *fc_output_data = reinterpret_cast<float *>(fc_output.buffer());
    for (size_t i = 0; i < output_shape[0]; ++i) {
        std::cout << "Output[" << i << "] = " << fc_output_data[i] << std::endl;
    }

    // Deallocate memory
    input.allocator()->free();
    fc_weights.allocator()->free();
    fc_biases.allocator()->free();
    fc_output.allocator()->free();

    return 0;
}


int main() {
        fc_data();
        std::cout << "----------------" << std::endl;
        fc_npy_data();
        return 0;
}
