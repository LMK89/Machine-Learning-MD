## Phương pháp nhân tử hóa ma trận và Tensor

Trang trình bày 1: Hệ số ma trận không âm (NMF)

NMF là một kỹ thuật mạnh mẽ để phân tích ma trận V không âm thành hai ma trận không âm W và H sao cho V ≈ WH. Phương pháp này được sử dụng rộng rãi trong việc giảm kích thước, trích xuất đặc trưng và nhận dạng mẫu.

```python
from sklearn.decomposition import NMF

# Generate a random nonnegative matrix
V = np.abs(np.random.randn(10, 5))

# Initialize NMF model
model = NMF(n_components=3, init='random', random_state=0)

# Fit the model and transform V
W = model.fit_transform(V)
H = model.components_

# Reconstruct the original matrix
V_approx = np.dot(W, H)

print("Original matrix shape:", V.shape)
print("W matrix shape:", W.shape)
print("H matrix shape:", H.shape)
print("Reconstructed matrix shape:", V_approx.shape)
```

Trang trình bày 2: Ví dụ thực tế về NMF: Phân tách hình ảnh

NMF có thể được sử dụng để phân tách hình ảnh thành các thành phần cơ bản, rất hữu ích trong việc nhận dạng khuôn mặt và xử lý hình ảnh.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_olivetti_faces

# Load Olivetti faces dataset
faces = fetch_olivetti_faces().data
n_samples, n_features = faces.shape

# Apply NMF
n_components = 10
model = NMF(n_components=n_components, init='random', random_state=0)
W = model.fit_transform(faces)
H = model.components_

# Plot original and reconstructed faces
fig, axes = plt.subplots(4, 5, figsize=(12, 8))
for i, ax in enumerate(axes.flatten()):
    if i < n_components:
        ax.imshow(H[i].reshape(64, 64), cmap=plt.cm.gray)
        ax.set_title(f'Component {i+1}')
    elif i < 2 * n_components:
        ax.imshow(faces[i-n_components].reshape(64, 64), cmap=plt.cm.gray)
        ax.set_title(f'Original {i-n_components+1}')
    else:
        reconstructed = np.dot(W[i-2*n_components], H)
        ax.imshow(reconstructed.reshape(64, 64), cmap=plt.cm.gray)
        ax.set_title(f'Reconstructed {i-2*n_components+1}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 3: Phương pháp Tensor

Các phương pháp tensor mở rộng các phép toán ma trận sang các mảng có chiều cao hơn, cho phép phân tích và biểu diễn dữ liệu phức tạp hơn. Những phương pháp này rất quan trọng trong các lĩnh vực như xử lý tín hiệu, thị giác máy tính và học máy.

```python
import tensorly as tl
from tensorly.decomposition import parafac

# Create a 3D tensor
tensor = np.random.rand(4, 5, 3)

# Perform CANDECOMP/PARAFAC (CP) decomposition
rank = 2
factors = parafac(tensor, rank=rank)

# Reconstruct the tensor
reconstructed_tensor = tl.kruskal_to_tensor(factors)

print("Original tensor shape:", tensor.shape)
print("Reconstructed tensor shape:", reconstructed_tensor.shape)
print("Reconstruction error:", np.linalg.norm(tensor - reconstructed_tensor))
```

Trang trình bày 4: Phương pháp Tensor: Phân tách Tucker

Phân rã Tucker là một phương pháp nhân tử tensor phổ biến khác, khái quát hóa SVD thành các tensor bậc cao hơn.

```python
import tensorly as tl
from tensorly.decomposition import tucker

# Create a 3D tensor
tensor = np.random.rand(4, 5, 3)

# Perform Tucker decomposition
core, factors = tucker(tensor, ranks=[2, 2, 2])

# Reconstruct the tensor
reconstructed_tensor = tl.tucker_to_tensor((core, factors))

print("Original tensor shape:", tensor.shape)
print("Core tensor shape:", core.shape)
print("Factor matrices shapes:", [f.shape for f in factors])
print("Reconstruction error:", np.linalg.norm(tensor - reconstructed_tensor))
```

Trang trình bày 5: Phục hồi thưa thớt

Phục hồi thưa thớt nhằm mục đích tái tạo lại các tín hiệu thưa thớt từ một số lượng nhỏ các phép đo tuyến tính. Kỹ thuật này là nền tảng trong cảm biến nén và xử lý tín hiệu.

```python
from sklearn.linear_model import Lasso

# Generate a sparse signal
n = 1000
k = 10
x = np.zeros(n)
x[np.random.choice(n, k, replace=False)] = np.random.randn(k)

# Create measurement matrix
m = 100
A = np.random.randn(m, n)

# Generate measurements
y = np.dot(A, x)

# Solve using Lasso (L1-regularized least squares)
lasso = Lasso(alpha=0.1)
x_recovered = lasso.fit(A, y).coef_

print("Original signal sparsity:", np.sum(x != 0))
print("Recovered signal sparsity:", np.sum(x_recovered != 0))
print("Recovery error:", np.linalg.norm(x - x_recovered) / np.linalg.norm(x))
```

Trang trình bày 6: Phục hồi thưa thớt: Theo đuổi kết hợp trực giao

Theo đuổi kết hợp trực giao (OMP) là một thuật toán tham lam để phục hồi thưa thớt, thường được sử dụng trong các ứng dụng cảm biến nén.

```python
from sklearn.linear_model import OrthogonalMatchingPursuit

# Generate a sparse signal
n = 1000
k = 10
x = np.zeros(n)
x[np.random.choice(n, k, replace=False)] = np.random.randn(k)

# Create measurement matrix
m = 100
A = np.random.randn(m, n)

# Generate measurements
y = np.dot(A, x)

# Solve using Orthogonal Matching Pursuit
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
x_recovered = omp.fit(A, y).coef_

print("Original signal sparsity:", np.sum(x != 0))
print("Recovered signal sparsity:", np.sum(x_recovered != 0))
print("Recovery error:", np.linalg.norm(x - x_recovered) / np.linalg.norm(x))
```

Slide 7: Học từ điển

Học từ điển liên quan đến việc tìm kiếm một cách biểu diễn thưa thớt của dữ liệu đầu vào theo một từ điển đã học. Kỹ thuật này rất hữu ích trong việc xử lý ảnh, trích xuất đặc trưng và nén.

```python
from sklearn.decomposition import DictionaryLearning
import matplotlib.pyplot as plt

# Generate random patches
n_samples, n_features = 1000, 64
data = np.random.randn(n_samples, n_features)

# Learn the dictionary
n_components = 100
dl = DictionaryLearning(n_components=n_components, alpha=1, max_iter=1000)
dictionary = dl.fit(data).components_

# Plot some dictionary atoms
fig, axes = plt.subplots(10, 10, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(dictionary[i].reshape(8, 8), cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 8: Học từ điển: Khử nhiễu hình ảnh

Học từ điển có thể được áp dụng cho việc khử nhiễu hình ảnh bằng cách học từ điển từ các mảng hình ảnh sạch và sử dụng nó để tái tạo lại các hình ảnh nhiễu.

```python
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import DictionaryLearning
from skimage import data, util
from skimage.restoration import denoise_dictionary_learning
import matplotlib.pyplot as plt

# Load and add noise to the image
image = util.img_as_float(data.camera())
noisy = image + 0.1 * np.random.randn(*image.shape)

# Extract patches and learn dictionary
patch_size = (8, 8)
patches = extract_patches_2d(noisy, patch_size)
dico = DictionaryLearning(n_components=100, alpha=1, max_iter=1000)
V = dico.fit(patches.reshape(len(patches), -1)).components_

# Denoise the image
denoised = denoise_dictionary_learning(noisy, dictionary=V.reshape((100, 8, 8)),
                                       patch_size=patch_size, alpha=0.1)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(noisy, cmap='gray')
axes[1].set_title('Noisy')
axes[2].imshow(denoised, cmap='gray')
axes[2].set_title('Denoised')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 9: Mô hình hỗn hợp Gaussian (GMM)

Mô hình hỗn hợp Gaussian là mô hình xác suất giả sử các điểm dữ liệu được tạo ra từ hỗn hợp của một số phân bố Gaussian hữu hạn. Chúng được sử dụng rộng rãi để phân cụm và ước tính mật độ.

```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
n_samples = 300
X = np.concatenate([
    np.random.normal(0, 1, (n_samples, 2)),
    np.random.normal(3, 1.5, (n_samples, 2)),
    np.random.normal(-2, 1, (n_samples, 2))
])

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(X)

# Plot results
x = np.linspace(-6, 6, 200)
y = np.linspace(-6, 6, 200)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X.shape)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c='white', s=10, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Negative log-likelihood')
plt.show()
```

Slide 10: Mô hình hỗn hợp Gaussian: Nhận dạng người nói

GMM có thể được sử dụng để nhận dạng người nói bằng cách lập mô hình phân bổ các đặc điểm âm thanh được trích xuất từ ​​tín hiệu giọng nói.

```python
from sklearn.mixture import GaussianMixture
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Simulated function to extract MFCC features from audio
def extract_mfcc(audio, n_mfcc=13):
    return np.random.randn(len(audio) // 1000, n_mfcc)

# Simulated function to load audio file
def load_audio(filename):
    return np.random.randn(44100 * 5)  # 5 seconds of audio at 44.1kHz

# Load and extract features from training data
speakers = ['speaker1', 'speaker2', 'speaker3']
models = {}

for speaker in speakers:
    audio = load_audio(f'{speaker}.wav')
    mfcc = extract_mfcc(audio)
    gmm = GaussianMixture(n_components=16, covariance_type='diag')
    models[speaker] = gmm.fit(mfcc)

# Test on new audio
test_audio = load_audio('test.wav')
test_mfcc = extract_mfcc(test_audio)

# Compute log-likelihood for each speaker
scores = {speaker: model.score(test_mfcc) for speaker, model in models.items()}

# Identify the speaker
identified_speaker = max(scores, key=scores.get)

print(f"Identified speaker: {identified_speaker}")
print("Log-likelihoods:")
for speaker, score in scores.items():
    print(f"{speaker}: {score}")
```

Slide 11: Hoàn thành ma trận

Hoàn thành ma trận là nhiệm vụ điền vào các mục còn thiếu của ma trận được quan sát một phần. Nó có các ứng dụng trong hệ thống gợi ý, vẽ hình ảnh và lọc cộng tác.

```python
from sklearn.impute import SimpleImputer

# Create a matrix with missing values
n, m = 10, 8
true_rank = 3
U = np.random.randn(n, true_rank)
V = np.random.randn(true_rank, m)
X = np.dot(U, V)

# Randomly mask some entries
mask = np.random.rand(n, m) < 0.3
X_incomplete = np.where(mask, np.nan, X)

# Perform matrix completion using simple mean imputation
imputer = SimpleImputer(strategy='mean')
X_completed = imputer.fit_transform(X_incomplete)

# Compute error
mse = np.mean((X - X_completed)**2)
print(f"Mean Squared Error: {mse}")

# Visualize results
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(X, cmap='viridis')
ax1.set_title('Original Matrix')
ax2.imshow(X_incomplete, cmap='viridis')
ax2.set_title('Incomplete Matrix')
ax3.imshow(X_completed, cmap='viridis')
ax3.set_title('Completed Matrix')
plt.show()
```

Trang trình bày 12: Hoàn thành ma trận: Lọc cộng tác

Hoàn thành ma trận thường được sử dụng trong lọc cộng tác cho các hệ thống đề xuất, chẳng hạn như dự đoán xếp hạng phim.

```python
from scipy.sparse.linalg import svds

# Create a user-item rating matrix with missing values
n_users, n_items = 100, 50
true_rank = 5
U = np.random.randn(n_users, true_rank)
V = np.random.randn(true_rank, n_items)
R = np.dot(U, V)

# Add some noise and mask random entries
R += 0.1 * np.random.randn(n_users, n_items)
mask = np.random.rand(n_users, n_items) < 0.8
R_observed = np.where(mask, R, 0)

# Perform matrix completion using SVD
U, s, Vt = svds(R_observed, k=true_rank)
S = np.diag(s)
R_completed = np.dot(np.dot(U, S), Vt)

# Compute error on observed entries
mse = np.mean((R[mask] - R_completed[mask])**2)
print(f"Mean Squared Error on observed entries: {mse}")

# Visualize results
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(R, cmap='viridis')
ax1.set_title('True Ratings')
ax2.imshow(R_observed, cmap='viridis')
ax2.set_title('Observed Ratings')
ax3.imshow(R_completed, cmap='viridis')
ax3.set_title('Predicted Ratings')
plt.show()
```

Trang trình bày 13: Tài nguyên bổ sung

Để khám phá thêm về các chủ đề được trình bày trong bài trình bày này, hãy xem xét các tài nguyên sau:

1. "Kỹ thuật nhân tố ma trận cho hệ thống gợi ý" của Koren và cộng sự. (2009) ArXiv: [https://arxiv.org/abs/0908.5614](https://arxiv.org/abs/0908.5614)
2. "Ứng dụng và phân rã tensor" của Kolda và Bader (2009) ArXiv: [https://arxiv.org/abs/0904.4505](https://arxiv.org/abs/0904.4505)
3. "Cảm biến nén" của Candès và Wakin (2008) ArXiv: [https://arxiv.org/abs/0801.2986](https://arxiv.org/abs/0801.2986)
4. "Thuật toán học từ điển để biểu diễn thưa thớt" của Tosic và Frossard (2011) ArXiv: [https://arxiv.org/abs/1009.2374](https://arxiv.org/abs/1009.2374)
5. "Hướng dẫn về các mô hình Markov ẩn và các ứng dụng được chọn trong nhận dạng giọng nói" của Rabiner (1989) Có sẵn tại: [https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20appluggest.pdf](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)

Những tài nguyên này cung cấp thông tin chuyên sâu về các chủ đề được thảo luận trong bài trình bày này và có thể đóng vai trò là điểm khởi đầu tuyệt vời cho việc nghiên cứu và nghiên cứu sâu hơn trong các lĩnh vực này.

Slide 14: Kết luận

Bài trình bày này đã đề cập đến một số chủ đề quan trọng trong học máy và xử lý tín hiệu:

1. Hệ số ma trận không âm (NMF)
2. Phương pháp tenxơ
3. Phục hồi thưa thớt
4. Học từ điển
5. Mô hình hỗn hợp Gaussian (GMM)
6. Hoàn thành ma trận

Những kỹ thuật này tạo thành nền tảng cho nhiều ứng dụng nâng cao trong phân tích dữ liệu, nhận dạng mẫu và xử lý tín hiệu. Bằng cách hiểu và áp dụng các phương pháp này, các nhà nghiên cứu và người thực hành có thể phát triển các công cụ mạnh mẽ để trích xuất thông tin có ý nghĩa từ các bộ dữ liệu phức tạp.

Khi lĩnh vực học máy tiếp tục phát triển, những kỹ thuật này có thể đóng vai trò ngày càng quan trọng trong việc giải quyết các vấn đề trong thế giới thực trên nhiều lĩnh vực khác nhau, bao gồm thị giác máy tính, xử lý ngôn ngữ tự nhiên, hệ thống gợi ý và nhiều lĩnh vực khác.

Chúng tôi khuyến khích bạn khám phá thêm các chủ đề này bằng cách sử dụng các tài nguyên được cung cấp và thử nghiệm triển khai các thuật toán này trong các dự án của riêng bạn. Hãy nhớ rằng các ví dụ về mã được cung cấp trong phần trình bày này nhằm mục đích minh họa các khái niệm cơ bản và các ứng dụng trong thế giới thực có thể yêu cầu triển khai và tối ưu hóa phức tạp hơn.
