from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff

# 두 레이블 이미지에서 Hausdorff Distance 계산
def calculate_hausdorff_distance(label_image_path1, label_image_path2):
    label1 = Image.open(label_image_path1).convert('L')
    label2 = Image.open(label_image_path2).convert('L')

    # 이미지를 numpy 배열로 변환
    label1_np = np.array(label1)
    label2_np = np.array(label2)

    # 엣지 위치 (값이 0이 아닌 위치)를 추출
    points1 = np.argwhere(label1_np > 0)
    points2 = np.argwhere(label2_np > 0)

    # directed_hausdorff 함수를 사용하여 양방향 Hausdorff 거리 계산
    hausdorff_1_2 = directed_hausdorff(points1, points2)[0]
    hausdorff_2_1 = directed_hausdorff(points2, points1)[0]

    # Hausdorff 거리 계산
    hausdorff_distance = max(hausdorff_1_2, hausdorff_2_1)
    return hausdorff_distance

# 두 레이블 이미지 파일 경로
label_image_path1 = 'training/1/label0_.jpg'
label_image_path2 = 'training/1/label1_.jpg'

# Hausdorff 거리 계산 및 출력
hausdorff_distance = calculate_hausdorff_distance(label_image_path1, label_image_path2)
print(f'Hausdorff Distance: {hausdorff_distance}')