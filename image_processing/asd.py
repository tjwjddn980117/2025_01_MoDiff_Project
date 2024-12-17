from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F

# 이미지 파일 열기
image_file_path = 'input_image.png'
image = Image.open(image_file_path).convert('L')  # 이미지를 그레이스케일로 변환
transform = T.ToTensor()
data = transform(image).squeeze(0)  # 텐서로 변환 및 채널 축 제거

data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

dilation = F.max_pool2d(data, kernel_size=2, stride=1, padding=1)
erosion = -F.max_pool2d(-data, kernel_size=2, stride=1, padding=1)

gradient = dilation - erosion
gradient = gradient.squeeze()  # 차원 축소 및 값 범위 조정
print(*gradient)
# 결과 이미지 저장
edge_image = T.ToPILImage()(gradient)
edge_image.save('output_edge_image.png')