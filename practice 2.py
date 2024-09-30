import numpy as np
import pandas as pd

# Đọc dữ liệu từ file CSV
file_path = r'C:\Users\acer\Desktop\Linear Practice\Practice2_Chapter2.csv'
data = pd.read_csv(file_path)

# Chuẩn bị dữ liệu cho thuật toán gradient descent
X = data[['X', 'Radio', 'Newspaper']].values  # Các biến đặc trưng (TV, Radio, Newspaper)
y = data['Sales'].values  # Biến mục tiêu (Sales - Doanh số)

# Thêm một cột toàn 1 vào X để sử dụng cho hệ số chặn (bias)
X = np.c_[np.ones(X.shape[0]), X]

# Chuẩn hóa các đặc trưng (trừ cột bias)
mean_X = np.mean(X[:, 1:], axis=0)
std_X = np.std(X[:, 1:], axis=0)
X[:, 1:] = (X[:, 1:] - mean_X) / std_X

# Khởi tạo tham số
theta = np.zeros(X.shape[1])  # Các hệ số (weights và bias)
alpha = 0.000001  # Tốc độ học (learning rate)
iterations = 1000  # Số lần lặp của gradient descent

# Định nghĩa hàm tính chi phí (cost function)
def compute_cost(X, y, theta):
    m = len(y)  # Số lượng mẫu
    predictions = X.dot(theta)  # Dự đoán giá trị
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)  # Tính chi phí
    return cost

# Thuật toán gradient descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)  # Số lượng mẫu
    cost_history = np.zeros(iterations)  # Lịch sử chi phí (để theo dõi quá trình giảm chi phí)
    
    for i in range(iterations):
        predictions = X.dot(theta)  # Dự đoán giá trị
        errors = predictions - y  # Sai số giữa dự đoán và thực tế
        gradient = (1 / m) * X.T.dot(errors)  # Gradient
        theta -= alpha * gradient  # Cập nhật các tham số
        cost_history[i] = compute_cost(X, y, theta)  # Lưu lại chi phí tại mỗi bước lặp
    
    return theta, cost_history

# Chạy thuật toán gradient descent
theta_final, cost_history = gradient_descent(X, y, theta, alpha, iterations)

# Xuất ra các hệ số cuối cùng và chi phí
final_cost = cost_history[-1]
final_parameters = pd.DataFrame({
    'Tham số': ['Hệ số chặn (bias)', 'Hệ số TV', 'Hệ số Radio', 'Hệ số Newspaper'],
    'Giá trị': theta_final
})

print("Các hệ số cuối cùng:\n", final_parameters)
print("\nChi phí cuối cùng:", final_cost)
