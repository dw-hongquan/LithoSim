'''
Author: Hongquan
Date: Mar. 24, 2025
Description: doinn/damo source-mask-dose-defocus-resist data pair
'''

import os
import cv2
# import numpy as np
from torch.utils.data import Dataset
import torch
from torch import Tensor
from typing import Tuple
from torchvision.transforms import transforms
import numpy as np

class LithoGenDataset(Dataset):
    def __init__(
        self,
        txt_path: str,
    ):
        super().__init__()

        with open(txt_path, 'r') as file:
            self.lines = file.readlines()
            file.close()

        self.transforms = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        line = self.lines[idx]
        litho_pair_path = line.split()
        source_path = litho_pair_path[0]
        mask_path = litho_pair_path[1]
        resist_path = litho_pair_path[4].strip()
        assert os.path.exists(source_path), ValueError(f'source path {source_path} not exit.')
        assert os.path.exists(mask_path), ValueError(f'mask path {mask_path} not exit.')
        assert os.path.exists(resist_path), ValueError(f'resist path {resist_path} not exit.')

        dose = torch.tensor(data = float(litho_pair_path[2])).unsqueeze(-1)
        defocus = torch.tensor(data = float(litho_pair_path[3]) / 80).unsqueeze(-1)

        mask_data = cv2.imread(mask_path, 0).astype(np.float32) / 255
        resist_data = cv2.imread(resist_path, 0).astype(np.float32) / 255

        mask_tensor = self.transforms(mask_data)
        resist_tensor = self.transforms(resist_data)

        # source [7180, 3]
        source_tensor = self.src2tensor(source_path)
        return source_tensor, mask_tensor, dose, defocus, resist_tensor

    def src2tensor(self, src_path: str) -> Tensor:
        data = []
        with open(src_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                row = [float(parts[0]), float(parts[1]), float(parts[2])]
                data.append(row)

        return torch.tensor(data)


if __name__ == "__main__":
    pass