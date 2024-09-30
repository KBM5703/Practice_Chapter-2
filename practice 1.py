import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu: Thời gian học (h) và Điểm thi
hours = np.array([155, 180, 164, 162, 181, 182, 173, 190, 171, 170, 181, 182, 189, 184, 209, 210])
scores = np.array([51, 52, 54, 53, 55, 59, 61, 59, 63, 76, 64, 66, 69, 72, 70, 80])

# Bước 1: Chuẩn bị tham số
m = 0  # hệ số góc ban đầu
b = 0  # hệ số chặn ban đầu
learning_rate = 0.0001  # Tốc độ học
epochs = 1000  # Số lần lặp

# Bước 2: Hàm tính hàm mất mát
def compute_loss(m, b, X, y):
    N = len(X)
    total_error = 0.0
    for i in range(N):
        total_error += (y[i] - (m * X[i] + b)) ** 2
    return total_error / N

# Bước 3: Hàm Gradient Descent
def gradient_descent(m, b, X, y, learning_rate, epochs):
    N = len(X)
    for _ in range(epochs):
        m_gradient = 0
        b_gradient = 0
        for i in range(N):
            x = X[i]
            y_actual = y[i]
            m_gradient += -(2/N) * x * (y_actual - (m * x + b))
            b_gradient += -(2/N) * (y_actual - (m * x + b))
        m -= learning_rate * m_gradient
        b -= learning_rate * b_gradient
    return m, b

# Bước 4: Huấn luyện mô hình
m, b = gradient_descent(m, b, hours, scores, learning_rate, epochs)

# Bước 5: Dự đoán
predicted_scores = m * hours + b

# Bước 6: Hiển thị kết quả
plt.scatter(hours, scores, color='blue', label='Dữ liệu thực tế')
plt.plot(hours, predicted_scores, color='red', label=f'Dự đoán (m={m:.2f}, b={b:.2f})')
plt.xlabel('Thời gian học (h)')
plt.ylabel('Điểm thi')
plt.legend()
plt.title('Hồi quy tuyến tính sử dụng Gradient Descent')
plt.show()

# In ra kết quả hệ số và hàm mất mát
print(f'Hệ số góc (m): {m:.4f}')
print(f'Hệ số chặn (b): {b:.4f}')
print(f'Hàm mất mát: {compute_loss(m, b, hours, scores):.4f}')
