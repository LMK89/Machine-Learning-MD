## Đệm hình ảnh và bước tiến hạt nhân trong mạng thần kinh chuyển đổi
Slide 1: Giới thiệu về Image Padding và Kernel Stride

Phần đệm hình ảnh và bước tiến của hạt nhân là những khái niệm quan trọng trong Mạng thần kinh chuyển đổi (CNN). Chúng đóng một vai trò quan trọng trong việc kiểm soát kích thước không gian của bản đồ tính năng đầu ra và trường tiếp nhận của mạng. Bài trình bày này sẽ khám phá những khái niệm này, cách triển khai và tác động của chúng đối với kiến ​​trúc CNN.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 5x5 image
image = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
])

# Display the image
plt.imshow(image, cmap='gray')
plt.title("Original 5x5 Image")
plt.show()
```

Trang trình bày 2: Đệm hình ảnh

Đệm hình ảnh liên quan đến việc thêm các pixel bổ sung xung quanh các cạnh của hình ảnh đầu vào. Kỹ thuật này được sử dụng để bảo toàn kích thước không gian của hình ảnh sau khi tích chập, cho phép tạo ra các mạng sâu hơn mà không cần giảm nhanh kích thước bản đồ đặc trưng.

```python
def pad_image(image, pad_width):
    return np.pad(image, pad_width, mode='constant', constant_values=0)

# Pad the image with a border of 1 pixel
padded_image = pad_image(image, 1)

plt.imshow(padded_image, cmap='gray')
plt.title("Padded 7x7 Image")
plt.show()
```

Slide 3: Các loại Padding

Có một số loại phần đệm, bao gồm phần đệm 0 (điền bằng số 0), phần đệm phản chiếu (phản chiếu các pixel cạnh) và phần đệm sao chép (nhập các pixel cạnh). Phần đệm bằng 0 là loại phổ biến nhất được sử dụng trong CNN.

```python
def pad_image_types(image, pad_width):
    zero_pad = np.pad(image, pad_width, mode='constant', constant_values=0)
    reflect_pad = np.pad(image, pad_width, mode='reflect')
    edge_pad = np.pad(image, pad_width, mode='edge')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(zero_pad, cmap='gray')
    ax1.set_title("Zero Padding")
    ax2.imshow(reflect_pad, cmap='gray')
    ax2.set_title("Reflection Padding")
    ax3.imshow(edge_pad, cmap='gray')
    ax3.set_title("Edge Padding")
    plt.show()

pad_image_types(image, 2)
```

Trang trình bày 4: Bước đi của hạt nhân

Bước tiến hạt nhân đề cập đến số lượng pixel mà bộ lọc tích chập di chuyển ở mỗi bước. Bước tiến bằng 1 có nghĩa là bộ lọc di chuyển từng pixel một, trong khi bước tiến lớn hơn dẫn đến kích thước đầu ra ít chồng chéo hơn và nhỏ hơn.

```python
def apply_convolution(image, kernel, stride):
    h, w = image.shape
    kh, kw = kernel.shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    output = np.zeros((oh, ow))

    for i in range(0, oh):
        for j in range(0, ow):
            output[i, j] = np.sum(image[i*stride:i*stride+kh, j*stride:j*stride+kw] * kernel)

    return output

kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
stride1_output = apply_convolution(image, kernel, 1)
stride2_output = apply_convolution(image, kernel, 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(stride1_output, cmap='gray')
ax1.set_title("Stride 1 Output")
ax2.imshow(stride2_output, cmap='gray')
ax2.set_title("Stride 2 Output")
plt.show()
```

Trang trình bày 5: Tác động của Sải bước đến kích thước đầu ra

Bước tiến ảnh hưởng đến kích thước không gian của bản đồ tính năng đầu ra. Bước tiến lớn hơn dẫn đến kích thước đầu ra nhỏ hơn, điều này có thể hữu ích để giảm độ phức tạp tính toán nhưng có thể dẫn đến mất thông tin không gian.

```python
def calculate_output_size(input_size, kernel_size, stride, padding):
    return ((input_size + 2 * padding - kernel_size) // stride) + 1

input_sizes = range(5, 51, 5)
strides = [1, 2, 3]
kernel_size = 3
padding = 1

plt.figure(figsize=(10, 6))
for stride in strides:
    output_sizes = [calculate_output_size(size, kernel_size, stride, padding) for size in input_sizes]
    plt.plot(input_sizes, output_sizes, label=f'Stride {stride}')

plt.xlabel('Input Size')
plt.ylabel('Output Size')
plt.title('Impact of Stride on Output Size')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 6: Đệm và Sải bước trong PyTorch

PyTorch, một khung học sâu phổ biến, cung cấp các chức năng tích hợp sẵn để áp dụng các phép tích chập với các cài đặt khoảng đệm và bước tiến khác nhau. Hãy xem cách sử dụng các tham số này trong lớp chập PyTorch.

```python
import torch
import torch.nn as nn

# Create a random 1x1x28x28 input tensor (batch_size x channels x height x width)
input_tensor = torch.randn(1, 1, 28, 28)

# Create convolutional layers with different padding and stride settings
conv_no_pad_stride1 = nn.Conv2d(1, 1, kernel_size=3, padding=0, stride=1)
conv_pad1_stride1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1)
conv_pad1_stride2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)

# Apply convolutions
output_no_pad_stride1 = conv_no_pad_stride1(input_tensor)
output_pad1_stride1 = conv_pad1_stride1(input_tensor)
output_pad1_stride2 = conv_pad1_stride2(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape (no padding, stride 1): {output_no_pad_stride1.shape}")
print(f"Output shape (padding 1, stride 1): {output_pad1_stride1.shape}")
print(f"Output shape (padding 1, stride 2): {output_pad1_stride2.shape}")
```

Trang trình bày 7: Ví dụ thực tế: Phát hiện cạnh

Phát hiện cạnh là một kỹ thuật xử lý hình ảnh cơ bản thường được sử dụng trong các tác vụ thị giác máy tính. Chúng ta có thể triển khai bộ lọc phát hiện cạnh đơn giản bằng cách sử dụng tích chập với phần đệm và bước nhảy thích hợp.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('path_to_your_image.jpg', 0)  # Load as grayscale
image = cv2.resize(image, (200, 200))  # Resize for demonstration

# Define edge detection kernels
horizontal_kernel = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]])

vertical_kernel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

# Apply convolution
horizontal_edges = cv2.filter2D(image, -1, horizontal_kernel)
vertical_edges = cv2.filter2D(image, -1, vertical_kernel)
combined_edges = cv2.addWeighted(horizontal_edges, 0.5, vertical_edges, 0.5, 0)

# Display results
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(horizontal_edges, cmap='gray')
ax2.set_title('Horizontal Edges')
ax3.imshow(vertical_edges, cmap='gray')
ax3.set_title('Vertical Edges')
ax4.imshow(combined_edges, cmap='gray')
ax4.set_title('Combined Edges')
plt.show()
```

Trang trình bày 8: Các kết cấu giãn nở

Các cuộn xoắn giãn, còn được gọi là các cuộn xoắn nhĩ, đưa ra một thông số khác gọi là tốc độ giãn nở. Điều này cho phép kernel bỏ qua các giá trị đầu vào, tăng trường tiếp nhận một cách hiệu quả mà không cần tăng số lượng tham số.

```python
def dilated_convolution(image, kernel, dilation):
    h, w = image.shape
    kh, kw = kernel.shape
    dkh, dkw = (kh-1) * dilation + 1, (kw-1) * dilation + 1
    oh, ow = h - dkh + 1, w - dkw + 1
    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            for ki in range(kh):
                for kj in range(kw):
                    ii = i + ki * dilation
                    jj = j + kj * dilation
                    output[i, j] += image[ii, jj] * kernel[ki, kj]

    return output

# Create a larger image for better visualization
larger_image = np.random.rand(15, 15)

kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

conv_normal = dilated_convolution(larger_image, kernel, dilation=1)
conv_dilated = dilated_convolution(larger_image, kernel, dilation=2)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(larger_image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(conv_normal, cmap='gray')
ax2.set_title('Normal Convolution')
ax3.imshow(conv_dilated, cmap='gray')
ax3.set_title('Dilated Convolution (rate=2)')
plt.show()
```

Trang trình bày 9: Chuyển đổi các kết cấu

Các phép tích chập chuyển đổi, đôi khi được gọi một cách không chính xác là các phép giải mã, được sử dụng để tăng kích thước không gian của đầu ra. Chúng thường được sử dụng trong các kiến ​​trúc bộ mã hóa-giải mã và các mô hình tổng quát.

```python
import torch
import torch.nn as nn

# Create a random 1x1x4x4 input tensor
input_tensor = torch.randn(1, 1, 4, 4)

# Create a transposed convolution layer
trans_conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)

# Apply transposed convolution
output = trans_conv(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")

# Visualize input and output
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(input_tensor.squeeze().detach().numpy(), cmap='gray')
ax1.set_title('Input')
ax2.imshow(output.squeeze().detach().numpy(), cmap='gray')
ax2.set_title('Output (Transposed Convolution)')
plt.show()
```

Slide 10: Trường tiếp nhận

Trường tiếp nhận đề cập đến vùng trong không gian đầu vào mà một tính năng CNN cụ thể đang xem xét. Khoảng đệm và bước tiến ảnh hưởng đến kích thước trường tiếp nhận, điều này rất quan trọng để hiểu những gì mạng "nhìn thấy" ở mỗi lớp.

```python
def calculate_receptive_field(num_layers, kernel_size, stride):
    receptive_field = kernel_size
    for _ in range(1, num_layers):
        receptive_field = receptive_field + (kernel_size - 1) * stride
    return receptive_field

num_layers = range(1, 6)
kernel_sizes = [3, 5, 7]
stride = 1

plt.figure(figsize=(10, 6))
for kernel_size in kernel_sizes:
    receptive_fields = [calculate_receptive_field(layers, kernel_size, stride) for layers in num_layers]
    plt.plot(num_layers, receptive_fields, marker='o', label=f'Kernel Size {kernel_size}')

plt.xlabel('Number of Layers')
plt.ylabel('Receptive Field Size')
plt.title('Growth of Receptive Field Size')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 11: Phần đệm và bước đi trong kiến ​​trúc CNN thực

Hãy xem xét cách sử dụng phần đệm và bước tiến trong các kiến ​​trúc CNN phổ biến như VGG16 và ResNet. Chúng tôi sẽ tạo một phiên bản đơn giản hóa của các mạng này để minh họa khái niệm này.

```python
import torch
import torch.nn as nn

class SimplifiedVGG(nn.Module):
    def __init__(self):
        super(SimplifiedVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

class SimplifiedResNet(nn.Module):
    def __init__(self):
        super(SimplifiedResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        identity = x
        x = self.residual_block(x)
        x += identity
        return self.relu(x)

# Create sample input
input_tensor = torch.randn(1, 3, 224, 224)

# Instantiate models
vgg = SimplifiedVGG()
resnet = SimplifiedResNet()

# Forward pass
vgg_output = vgg.features(input_tensor)
resnet_output = resnet(input_tensor)

print(f"VGG Input shape: {input_tensor.shape}")
print(f"VGG Output shape: {vgg_output.shape}")
print(f"ResNet Input shape: {input_tensor.shape}")
print(f"ResNet Output shape: {resnet_output.shape}")
```

Slide 12: Ví dụ thực tế: Phân đoạn hình ảnh

Phân đoạn hình ảnh là một nhiệm vụ trong đó phần đệm và bước tiến đóng vai trò quan trọng. Hãy triển khai một kiến ​​trúc đơn giản giống như U-Net để phân đoạn hình ảnh, thể hiện việc sử dụng các giá trị đệm và sải bước khác nhau.

```python
import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # Encoder (downsampling)
        self.enc1 = self.conv_block(3, 64, padding=1)
        self.enc2 = self.conv_block(64, 128, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Bridge
        self.bridge = self.conv_block(128, 256, padding=1)

        # Decoder (upsampling)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(256, 128, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64, padding=1)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch, padding):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))

        # Bridge
        bridge = self.bridge(self.pool(enc2))

        # Decoding
        dec1 = self.dec1(torch.cat([self.upconv1(bridge), enc2], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec1), enc1], dim=1))

        return self.final(dec2)

# Create a sample input
input_tensor = torch.randn(1, 3, 256, 256)

# Instantiate the model
model = SimpleUNet()

# Forward pass
output = model(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
```

Slide 13: Lựa chọn đệm và bước đi phù hợp

Việc lựa chọn khoảng đệm và bước tiến phụ thuộc vào nhiệm vụ cụ thể và kiến ​​trúc mạng. Dưới đây là một số hướng dẫn chung:

1. Sử dụng phần đệm 'giống nhau' (phần đệm giữ kích thước không gian không đổi) cho các mạng sâu hơn để ngăn chặn việc giảm nhanh kích thước bản đồ đặc điểm.
2. Sử dụng các bước tiến lớn hơn ở các lớp trước đó để giảm kích thước không gian và chi phí tính toán.
3. Trong các nhiệm vụ yêu cầu thông tin không gian chi tiết (ví dụ: phân đoạn), hãy sử dụng các bước nhỏ hơn và duy trì các kích thước không gian.
4. Đối với các nhiệm vụ phân loại, việc giảm dần các kích thước không gian thường có lợi.

```python
def calculate_output_size(input_size, kernel_size, stride, padding):
    return ((input_size + 2 * padding - kernel_size) // stride) + 1

def print_layer_info(layer_name, input_size, kernel_size, stride, padding):
    output_size = calculate_output_size(input_size, kernel_size, stride, padding)
    print(f"{layer_name}: Input={input_size}, Output={output_size}")

# Example network configuration
input_size = 224
print_layer_info("Conv1", input_size, kernel_size=7, stride=2, padding=3)
input_size = calculate_output_size(input_size, 7, 2, 3)
print_layer_info("MaxPool", input_size, kernel_size=3, stride=2, padding=1)
input_size = calculate_output_size(input_size, 3, 2, 1)
print_layer_info("Conv2", input_size, kernel_size=3, stride=1, padding=1)
input_size = calculate_output_size(input_size, 3, 1, 1)
print_layer_info("Conv3", input_size, kernel_size=3, stride=2, padding=1)
```

Trang trình bày 14: Tối ưu hóa phần đệm và bước chạy để đạt hiệu suất

Việc sử dụng phần đệm và bước chạy phù hợp có thể tác động đáng kể đến hiệu suất và hiệu quả của CNN:

1. Kích thước không gian giảm (bước tiến lớn hơn) làm giảm độ phức tạp tính toán nhưng có thể mất thông tin chi tiết.
2. Duy trì kích thước không gian (khoảng đệm thích hợp) cho phép mạng sâu hơn nhưng làm tăng chi phí tính toán.
3. Việc sử dụng phần đệm có thể giúp lưu giữ thông tin ở các cạnh của đầu vào, điều này rất quan trọng đối với các tác vụ như phát hiện đối tượng.
4. Stride có thể được sử dụng như một giải pháp thay thế cho các lớp gộp để lấy mẫu xuống, có khả năng làm giảm số lượng tham số.

```python
import time

def benchmark_conv(input_size, kernel_size, stride, padding, iterations=1000):
    input_tensor = torch.randn(1, 3, input_size, input_size)
    conv_layer = nn.Conv2d(3, 64, kernel_size=kernel_size, stride=stride, padding=padding)

    start_time = time.time()
    for _ in range(iterations):
        _ = conv_layer(input_tensor)
    end_time = time.time()

    return end_time - start_time

# Benchmark different configurations
configs = [
    {"name": "No padding, stride 1", "kernel": 3, "stride": 1, "padding": 0},
    {"name": "With padding, stride 1", "kernel": 3, "stride": 1, "padding": 1},
    {"name": "No padding, stride 2", "kernel": 3, "stride": 2, "padding": 0},
    {"name": "With padding, stride 2", "kernel": 3, "stride": 2, "padding": 1},
]

for config in configs:
    time_taken = benchmark_conv(224, config["kernel"], config["stride"], config["padding"])
    print(f"{config['name']}: {time_taken:.4f} seconds")
```

Trang trình bày 15: Tài nguyên bổ sung

Để biết thêm thông tin chuyên sâu về phần đệm hình ảnh và bước tiến của nhân trong Mạng thần kinh chuyển đổi, hãy xem xét khám phá các tài nguyên sau:

1. Bài viết ArXiv: "Hướng dẫn về số học tích chập để học sâu" của Vincent Dumoulin và Francesco Visin ([https://arxiv.org/abs/1603.07285](https://arxiv.org/abs/1603.07285))
2. Bài viết ArXiv: "Deconvolution and Checkerboard Artifacts" của Augustus Odena, Vincent Dumoulin và Chris Olah ([https://arxiv.org/abs/1611.07308](https://arxiv.org/abs/1611.07308))
3. Sách Deep Learning của Ian Goodfellow, Yoshua Bengio và Aaron Courville, đặc biệt là Chương 9 về Mạng chuyển đổi ([https://www.deeplearningbook.org/](https://www.deeplearningbook.org/))

Những tài nguyên này cung cấp những giải thích toàn diện và nền tảng toán học cho các khái niệm được thảo luận trong bài trình bày này.
