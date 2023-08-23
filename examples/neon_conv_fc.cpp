// Ulrik Hørlyk Hjort
// Two layer network example conv and fully connected

#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/Tensor.h"

using namespace arm_compute;

int main() {


    // Define tensor shapes
    TensorShape input_shape(4, 4, 1);           // Input image of size 4x4x1 (width, height, channels)
    TensorShape conv_weights_shape(3, 3, 1, 1); // Convolution weights shape (width, height, input channels, output channels)
    TensorShape conv_biases_shape(1);          // Convolution biases shape (output channels)
    TensorShape conv_output_shape(2, 2, 1);    // Convolution output feature map of size 2x2x1
    TensorShape fc_weights_shape(4, 1, 1, 1);  // Fully connected weights shape (input size, output size)
    TensorShape fc_biases_shape(1);            // Fully connected biases shape (output size)
    TensorShape fc_output_shape(1);            // Fully connected output size

    // Create tensors
    Tensor input;
    Tensor conv_weights;
    Tensor conv_biases;
    Tensor conv_output;
    Tensor fc_weights;
    Tensor fc_biases;
    Tensor fc_output;

    input.allocator()->init(TensorInfo(input_shape, 1, DataType::F32));
    conv_weights.allocator()->init(TensorInfo(conv_weights_shape, 1, DataType::F32));
    conv_biases.allocator()->init(TensorInfo(conv_biases_shape, 1, DataType::F32));
    conv_output.allocator()->init(TensorInfo(conv_output_shape, 1, DataType::F32));
    fc_weights.allocator()->init(TensorInfo(fc_weights_shape, 1, DataType::F32));
    fc_biases.allocator()->init(TensorInfo(fc_biases_shape, 1, DataType::F32));
    fc_output.allocator()->init(TensorInfo(fc_output_shape, 1, DataType::F32));

    // Allocate memory for tensors
    input.allocator()->allocate();
    conv_weights.allocator()->allocate();
    conv_biases.allocator()->allocate();
    conv_output.allocator()->allocate();
    fc_weights.allocator()->allocate();
    fc_biases.allocator()->allocate();
    fc_output.allocator()->allocate();

    // Load custom data into tensors
    float custom_input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float custom_conv_weights[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    float custom_conv_biases[] = {0};
    float custom_fc_weights[] = {0.5, -0.5, -1.0, 1.0};
    float custom_fc_biases[] = {0.25};

    std::memcpy(input.buffer(), custom_input, sizeof(custom_input));
    std::memcpy(conv_weights.buffer(), custom_conv_weights, sizeof(custom_conv_weights));
    std::memcpy(conv_biases.buffer(), custom_conv_biases, sizeof(custom_conv_biases));
    std::memcpy(fc_weights.buffer(), custom_fc_weights, sizeof(custom_fc_weights));
    std::memcpy(fc_biases.buffer(), custom_fc_biases, sizeof(custom_fc_biases));

    // Configure and run the convolution layer
    NEConvolutionLayer convolution;
    PadStrideInfo conv_info(1, 1, 0, 0);
    convolution.configure(&input, &conv_weights, &conv_biases, &conv_output, conv_info);
    convolution.run();

    // Configure and run the fully connected layer
    NEFullyConnectedLayer fully_connected;
    fully_connected.configure(&conv_output, &fc_weights, &fc_biases, &fc_output);
    fully_connected.run();

    // Print the output of the fully connected layer
    float *fc_output_data = reinterpret_cast<float *>(fc_output.buffer());
    std::cout << "Output of Fully Connected Layer: " << fc_output_data[0] << std::endl;

    // Deallocate memory
    input.allocator()->free();
    conv_weights.allocator()->free();
    conv_biases.allocator()->free();
    conv_output.allocator()->free();
    fc_weights.allocator()->free();
    fc_biases.allocator()->free();
    fc_output.allocator()->free();

    return 0;
}
