import numpy as np
import matplotlib.pyplot as plt

# NumPy 파일 불러오기
file_path = './log/lidc/run/test/weighted_4_mean_metrics.npy'  # 파일 경로를 설정하세요.
mean_metrics = np.load(file_path)

# 데이터 확인
print(f"Data shape: {mean_metrics.shape}")
print(mean_metrics)