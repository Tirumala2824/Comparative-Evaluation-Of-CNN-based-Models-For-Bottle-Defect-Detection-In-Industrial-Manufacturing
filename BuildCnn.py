import numpy as np

class CustomConv2D:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.filters = np.random.randn(output_channels, input_channels, kernel_size, kernel_size)
        self.bias = np.zeros((output_channels, 1))
        self.stride = stride
        self.padding = padding

    def forward(self, input_volume):
        batch_size, input_channels, input_height, input_width = input_volume.shape
        output_channels, _, kernel_size, _ = self.filters.shape

        # Calculate output dimensions
        output_height = (input_height - kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (input_width - kernel_size + 2 * self.padding) // self.stride + 1

        # Apply padding to input volume
        padded_input = np.pad(input_volume, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        # Initialize output volume
        output_volume = np.zeros((batch_size, output_channels, output_height, output_width))

        # Perform convolution
        for b in range(batch_size):
            for c_out in range(output_channels):
                for h_out in range(output_height):
                    for w_out in range(output_width):
                        h_start = h_out * self.stride
                        h_end = h_start + kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + kernel_size

                        receptive_field = padded_input[b, :, h_start:h_end, w_start:w_end]
                        output_volume[b, c_out, h_out, w_out] = np.sum(receptive_field * self.filters[c_out]) + self.bias[c_out]

        return output_volume


class CustomMaxPool2D:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_volume):
        batch_size, input_channels, input_height, input_width = input_volume.shape

        # Calculate output dimensions
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1

        # Initialize output volume
        output_volume = np.zeros((batch_size, input_channels, output_height, output_width))

        # Perform max pooling
        for b in range(batch_size):
            for c in range(input_channels):
                for h_out in range(output_height):
                    for w_out in range(output_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.pool_size

                        receptive_field = input_volume[b, c, h_start:h_end, w_start:w_end]
                        output_volume[b, c, h_out, w_out] = np.max(receptive_field)

        return output_volume

# Example usage:
input_shape = (3, 32, 32)  # Channels, Height, Width
num_classes = 10
x = np.random.randn(2, *input_shape)  # Sample input batch

# Convolutional layer
conv_layer = CustomConv2D(input_channels=3, output_channels=6, kernel_size=3, stride=1, padding=1)
conv_output = conv_layer.forward(x)

# Max pooling layer
pool_layer = CustomMaxPool2D(pool_size=2, stride=2)
pool_output = pool_layer.forward(conv_output)
