## Kỹ thuật tăng cường dữ liệu cho mạng nơ-ron tích chập
Slide 1: Tăng cường dữ liệu trong CNN

Tăng cường dữ liệu là một kỹ thuật mạnh mẽ được sử dụng để tăng tính đa dạng của dữ liệu huấn luyện cho mạng thần kinh tích chập (CNN). Nó liên quan đến việc tạo các mẫu đào tạo mới bằng cách áp dụng các phép biến đổi khác nhau cho các hình ảnh hiện có. Quá trình này giúp cải thiện khả năng khái quát hóa mô hình và giảm tình trạng trang bị quá mức, đặc biệt khi làm việc với các tập dữ liệu hạn chế.

```python
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

# Load MNIST dataset
(X_train, _), (_, _) = mnist.load_data()

# Select a sample image
sample_image = X_train[0]

# Create an ImageDataGenerator instance
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Reshape the image to (1, 28, 28, 1)
sample_image = sample_image.reshape((1, 28, 28, 1))

# Generate augmented images
augmented_images = [sample_image]
for _ in range(5):
    augmented_images.append(datagen.flow(sample_image, batch_size=1)[0])

# Plot original and augmented images
plt.figure(figsize=(10, 2))
for i, img in enumerate(augmented_images):
    plt.subplot(1, 6, i+1)
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title('Original')
    else:
        plt.title(f'Augmented {i}')
plt.tight_layout()
plt.show()
```

Trang trình bày 2: Xoay ảnh

Xoay là một kỹ thuật tăng cường phổ biến liên quan đến việc xoay hình ảnh theo một góc ngẫu nhiên trong phạm vi xác định. Điều này giúp mô hình trở nên bất biến với hướng của các đối tượng trong ảnh.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

# Load a sample image
image = cv2.imread('sample_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate rotated images
angles = [0, 45, 90, 135, 180]
rotated_images = [rotate_image(image, angle) for angle in angles]

# Display the results
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, (img, angle) in enumerate(zip(rotated_images, angles)):
    axes[i].imshow(img)
    axes[i].set_title(f'Rotation: {angle}°')
    axes[i].axis('off')
plt.tight_layout()
plt.show()
```

Slide 3: Lật ngang và lật dọc

Lật là một kỹ thuật tăng cường hiệu quả khác giúp tạo ra hình ảnh phản chiếu của dữ liệu gốc. Điều này đặc biệt hữu ích cho các đối tượng có thể xuất hiện theo các hướng khác nhau.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)

# Load a sample image
image = cv2.imread('sample_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate flipped images
flipped_horizontal = flip_image(image, 1)
flipped_vertical = flip_image(image, 0)
flipped_both = flip_image(image, -1)

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(image)
axes[0, 0].set_title('Original')
axes[0, 1].imshow(flipped_horizontal)
axes[0, 1].set_title('Horizontal Flip')
axes[1, 0].imshow(flipped_vertical)
axes[1, 0].set_title('Vertical Flip')
axes[1, 1].imshow(flipped_both)
axes[1, 1].set_title('Both Flips')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

Slide 4: Cắt ngẫu nhiên

Cắt ngẫu nhiên bao gồm việc chọn một phần ngẫu nhiên của hình ảnh và sử dụng nó làm mẫu huấn luyện mới. Kỹ thuật này giúp mô hình tập trung vào các phần khác nhau của hình ảnh và trở nên chắc chắn hơn đối với hiện tượng che khuất một phần.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]
    return crop

# Load a sample image
image = cv2.imread('sample_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate random crops
crops = [random_crop(image, 200, 200) for _ in range(4)]

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, crop in enumerate(crops):
    row = i // 2
    col = i % 2
    axes[row, col].imshow(crop)
    axes[row, col].set_title(f'Random Crop {i+1}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Trang trình bày 5: Hiện tượng nhiễu màu

Hiện tượng biến đổi màu sắc liên quan đến việc thay đổi ngẫu nhiên độ sáng, độ tương phản, độ bão hòa và màu sắc của hình ảnh. Kỹ thuật này giúp mô hình trở nên chắc chắn hơn trước những thay đổi trong điều kiện ánh sáng và cân bằng màu sắc.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_jitter(image, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Jitter brightness
    brightness_factor = 1.0 + np.random.uniform(-brightness, brightness)
    image[:,:,2] = np.clip(image[:,:,2] * brightness_factor, 0, 255)

    # Jitter contrast
    contrast_factor = 1.0 + np.random.uniform(-contrast, contrast)
    image[:,:,2] = np.clip(((image[:,:,2] - 128) * contrast_factor) + 128, 0, 255)

    # Jitter saturation
    saturation_factor = 1.0 + np.random.uniform(-saturation, saturation)
    image[:,:,1] = np.clip(image[:,:,1] * saturation_factor, 0, 255)

    # Jitter hue
    hue_factor = np.random.uniform(-hue, hue)
    image[:,:,0] = (image[:,:,0] + hue_factor * 180) % 180

    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image.astype(np.uint8)

# Load a sample image
image = cv2.imread('sample_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate color jittered images
jittered_images = [color_jitter(image) for _ in range(4)]

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, img in enumerate(jittered_images):
    row = i // 2
    col = i % 2
    axes[row, col].imshow(img)
    axes[row, col].set_title(f'Color Jittered {i+1}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Trang trình bày 6: Bổ sung nhiễu Gaussian

Việc thêm nhiễu Gaussian vào hình ảnh có thể giúp cải thiện độ bền của mô hình đối với nhiễu trong các tình huống thực tế. Kỹ thuật này mô phỏng sự không hoàn hảo trong việc chụp hoặc truyền hình ảnh.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

# Load a sample image
image = cv2.imread('sample_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate noisy images with different standard deviations
std_devs = [10, 25, 50]
noisy_images = [add_gaussian_noise(image, std=std) for std in std_devs]

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(image)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for i, (img, std) in enumerate(zip(noisy_images, std_devs)):
    row = (i + 1) // 2
    col = (i + 1) % 2
    axes[row, col].imshow(img)
    axes[row, col].set_title(f'Noise std: {std}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Slide 7: Biến dạng đàn hồi

Biến dạng đàn hồi là một kỹ thuật tăng cường nâng cao áp dụng các phép biến đổi phi tuyến tính cho hình ảnh. Điều này đặc biệt hữu ích cho các tác vụ nhận dạng chữ số viết tay vì nó mô phỏng các biến thể tự nhiên trong chữ viết tay.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)

# Load a sample image (assuming it's a grayscale image)
image = cv2.imread('sample_digit.png', 0)

# Generate elastically deformed images
alphas = [10, 30, 50]
sigma = 5
deformed_images = [elastic_transform(image, alpha, sigma) for alpha in alphas]

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for i, (img, alpha) in enumerate(zip(deformed_images, alphas)):
    row = (i + 1) // 2
    col = (i + 1) % 2
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(f'Alpha: {alpha}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Slide 8: Hỗn hợp

Mixup là một kỹ thuật tăng cường dữ liệu nhằm tạo ra các mẫu huấn luyện mới bằng cách nội suy tuyến tính giữa các cặp hình ảnh và nhãn của chúng. Điều này giúp mô hình tìm hiểu các ranh giới quyết định mượt mà hơn và cải thiện tính khái quát hóa.

```python
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

def mixup(x1, x2, y1, y2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y

# Load CIFAR-10 dataset
(x_train, y_train), (_, _) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0

# Select two random images
idx1, idx2 = np.random.randint(0, len(x_train), 2)
img1, img2 = x_train[idx1], x_train[idx2]
label1, label2 = y_train[idx1], y_train[idx2]

# Apply mixup
mixed_img, mixed_label = mixup(img1, img2, label1, label2)

# Display the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img1)
axes[0].set_title(f'Image 1 (Label: {label1[0]})')
axes[0].axis('off')

axes[1].imshow(img2)
axes[1].set_title(f'Image 2 (Label: {label2[0]})')
axes[1].axis('off')

axes[2].imshow(mixed_img)
axes[2].set_title(f'Mixed Image (Label: {mixed_label[0]:.2f}, {1-mixed_label[0]:.2f})')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

Trang trình bày 9: Xóa ngẫu nhiên

Xóa ngẫu nhiên là một kỹ thuật tăng cường chọn ngẫu nhiên các vùng hình chữ nhật trong ảnh và thay thế chúng bằng nhiễu ngẫu nhiên hoặc giá trị không đổi. Điều này giúp mô hình trở nên chắc chắn hơn đối với các phần bị che khuất và thiếu trong hình ảnh.

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def random_erasing(image, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=1/0.3):
    if np.random.rand() > p:
        return image

    h, w, c = image.shape
    s = np.random.uniform(sl, sh) * h * w
    r = np.random.uniform(r1, r2)

    new_h = int(np.sqrt(s / r))
    new_w = int(np.sqrt(s * r))

    left = np.random.randint(0, w - new_w)
    top = np.random.randint(0, h - new_h)

    erased_area = image[top:top+new_h, left:left+new_w, :]
    erased_area[:] = np.random.randint(0, 256, size=erased_area.shape)

    return image

# Load a sample image
image = np.array(Image.open('sample_image.jpg'))

# Apply random erasing multiple times
erased_images = [random_erasing(image.()) for _ in range(4)]

# Display the results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(image)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for i, img in enumerate(erased_images):
    row = i // 3
    col = i % 3 + 1 if i < 3 else i % 3
    axes[row, col].imshow(img)
    axes[row, col].set_title(f'Erased {i+1}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Trang trình bày 10: Cắt bỏ

Cutout là một kỹ thuật tăng cường dữ liệu đơn giản nhưng hiệu quả, bao gồm việc che giấu ngẫu nhiên các vùng hình vuông của hình ảnh đầu vào. Điều này khuyến khích người mẫu tập trung vào toàn bộ đối tượng trong ảnh, thay vì dựa vào các đặc điểm cụ thể.

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def cutout(image, n_holes=1, length=50):
    h, w = image.shape[:2]
    mask = np.ones((h, w), np.float32)

    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0

    masked_image = image.()
    masked_image[:,:,0] = image[:,:,0] * mask
    masked_image[:,:,1] = image[:,:,1] * mask
    masked_image[:,:,2] = image[:,:,2] * mask

    return masked_image

# Load a sample image
image = np.array(Image.open('sample_image.jpg'))

# Apply cutout with different parameters
cutout_images = [
    cutout(image.(), n_holes=1, length=50),
    cutout(image.(), n_holes=2, length=40),
    cutout(image.(), n_holes=3, length=30)
]

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(image)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for i, img in enumerate(cutout_images):
    row = (i + 1) // 2
    col = (i + 1) % 2
    axes[row, col].imshow(img)
    axes[row, col].set_title(f'Cutout {i+1}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Trang trình bày 11: CutMix

CutMix là một kỹ thuật tăng cường dữ liệu nâng cao kết hợp các khía cạnh của cả Mixup và Cutout. Nó liên quan đến việc cắt và dán các bản vá từ hình ảnh đào tạo này sang hình ảnh đào tạo khác, điều chỉnh nhãn tương ứng với diện tích của bản vá.

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def cutmix(image1, image2, alpha=1.0):
    h, w = image1.shape[:2]

    # Generate random bounding box
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    # Create mixed image
    mixed_image = image1.()
    mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]

    # Adjust lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

    return mixed_image, lam

# Load two sample images
image1 = np.array(Image.open('sample_image1.jpg'))
image2 = np.array(Image.open('sample_image2.jpg'))

# Apply CutMix
mixed_image, lam = cutmix(image1, image2)

# Display the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image1)
axes[0].set_title('Image 1')
axes[0].axis('off')

axes[1].imshow(image2)
axes[1].set_title('Image 2')
axes[1].axis('off')

axes[2].imshow(mixed_image)
axes[2].set_title(f'CutMix (λ = {lam:.2f})')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

Trang trình chiếu 12: Ví dụ thực tế: Nhận dạng nét mặt

Tăng cường dữ liệu là rất quan trọng trong các nhiệm vụ nhận dạng biểu cảm khuôn mặt để cải thiện hiệu suất và tính tổng quát của mô hình. Dưới đây là ví dụ về cách áp dụng các kỹ thuật tăng cường khác nhau cho tập dữ liệu biểu cảm khuôn mặt.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))
    return image

# Load a sample facial expression image
image = load_and_preprocess_image('sample_face.jpg')

# Create an ImageDataGenerator instance
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate augmented images
augmented_images = [image]
for _ in range(5):
    augmented_images.append(datagen.random_transform(image))

# Display the results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, img in enumerate(augmented_images):
    row = i // 3
    col = i % 3
    axes[row, col].imshow(img)
    axes[row, col].set_title('Original' if i == 0 else f'Augmented {i}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Trang trình chiếu 13: Ví dụ thực tế: Phát hiện bệnh cây

Tăng cường dữ liệu đóng một vai trò quan trọng trong việc cải thiện các mô hình phát hiện bệnh cây trồng, đặc biệt là khi xử lý các bộ dữ liệu hạn chế. Dưới đây là ví dụ về cách áp dụng các kỹ thuật tăng cường khác nhau cho hình ảnh lá cây để phân loại bệnh.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return image

# Load a sample plant leaf image
image = load_and_preprocess_image('sample_leaf.jpg')

# Create an ImageDataGenerator instance
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Generate augmented images
augmented_images = [image]
for _ in range(5):
    augmented_images.append(datagen.random_transform(image))

# Display the results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, img in enumerate(augmented_images):
    row = i // 3
    col = i % 3
    axes[row, col].imshow(img)
    axes[row, col].set_title('Original' if i == 0 else f'Augmented {i}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

Để biết thêm thông tin chuyên sâu về các kỹ thuật tăng cường dữ liệu và ứng dụng của chúng trong CNN, hãy xem xét khám phá các tài liệu nghiên cứu sau:

1. "Khảo sát về Tăng cường dữ liệu hình ảnh cho học sâu" của Connor Shorten và Taghi M. Khoshgoftaar (2019) ArXiv: [https://arxiv.org/abs/1912.11899](https://arxiv.org/abs/1912.11899)
2. "Tự động tăng cường: Học các chiến lược tăng cường từ dữ liệu" của Ekin D. Cubuk và cộng sự. (2018) ArXiv: [https://arxiv.org/abs/1805.09501](https://arxiv.org/abs/1805.09501)
3. "RandAugment: Tăng cường dữ liệu tự động thực tế với không gian tìm kiếm giảm" của Ekin D. Cubuk et al. (2019) ArXiv: [https://arxiv.org/abs/1909.13719](https://arxiv.org/abs/1909.13719)
4. "Cải thiện việc chính quy hóa các mạng thần kinh phức tạp có phần cắt bỏ" của Terrance DeVries và Graham W. Taylor (2017) ArXiv: [https://arxiv.org/abs/1708.04552](https://arxiv.org/abs/1708.04552)
5. "CutMix: Chiến lược chính quy hóa để đào tạo các bộ phân loại mạnh với các tính năng có thể bản địa hóa" của Sangdoo Yun và cộng sự. (2019) ArXiv: [https://arxiv.org/abs/1905.04899](https://arxiv.org/abs/1905.04899)

Những tài nguyên này cung cấp những hiểu biết sâu sắc có giá trị về những tiến bộ mới nhất trong kỹ thuật tăng cường dữ liệu cho CNN và tác động của chúng đối với hiệu suất mô hình.
