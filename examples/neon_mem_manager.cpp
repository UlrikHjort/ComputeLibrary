#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/core/Utils.h"

using namespace arm_compute;

int main() {


    // Define tensor shapes
    TensorShape input_shape(4, 1, 1);  // Input tensor shape (batch size, input size, channels)
    TensorShape fc_weights_shape(1, 4, 1, 1);  // Fully connected weights shape (input size, output size)
    TensorShape fc_biases_shape(1);  // Fully connected biases shape (output size)
    TensorShape fc_output_shape(1);  // Fully connected output size

    // Create tensors
    Tensor input;
    Tensor fc_weights;
    Tensor fc_biases;
    Tensor fc_output;

    // Create BlobLifetimeManager and PoolManager
    BlobLifetimeManager blob_manager;
    PoolManager pool_manager;

    // Initialize tensor with BlobLifetimeManager and PoolManager
    input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32), &pool_manager, &blob_manager);
    fc_weights.allocator()->init(TensorInfo(fc_weights_shape, 1, DataType::F32), &pool_manager, &blob_manager);
    fc_biases.allocator()->init(TensorInfo(fc_biases_shape, 1, DataType::F32), &pool_manager, &blob_manager);
    fc_output.allocator()->init(TensorInfo(fc_output_shape, 1, DataType::F32), &pool_manager, &blob_manager);

    // Allocate memory for tensors
    input.allocator()->allocate();
    fc_weights.allocator()->allocate();
    fc_biases.allocator()->allocate();
    fc_output.allocator()->allocate();

    // Generate input data (example values)
    float input_data[] = {1.0, 2.0, 3.0, 4.0};
    std::memcpy(input.buffer(), input_data, sizeof(input_data));

    // Manually set weights and biases (example values)
    float weights_data[] = {0.5, -0.5, -1.0, 1.0};
    float biases_data[] = {0.25};

    std::memcpy(fc_weights.buffer(), weights_data, sizeof(weights_data));
    std::memcpy(fc_biases.buffer(), biases_data, sizeof(biases_data));

    // Configure and run the fully connected layer
    NEFullyConnectedLayer fully_connected;
    fully_connected.configure(&input, &fc_weights, &fc_biases, &fc_output);
    fully_connected.run();

    // Print the output of the fully connected layer
    float *fc_output_data = reinterpret_cast<float *>(fc_output.buffer());
    std::cout << "Output of Fully Connected Layer: " << fc_output_data[0] << std::endl;

    // Deallocate memory (Not necessary with BlobLifetimeManager)
    // input.allocator()->free();
    // fc_weights.allocator()->free();
    // fc_biases.allocator()->free();
    // fc_output.allocator()->free();

    return 0;
}
