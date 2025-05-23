'''
Author: Hongquan
Date: Feb. 26, 2025
Description: a tool to generate your own dataset path, please replace ``dataset_dir`` and ``txt_path`` to true path for your dataset.
'''
import os

def dataset2txt_fast(
        dataset_dir: str = './lithosim/opc_metal/train_val',
        txt_path: str = './lithosim_path/opc_metal'
        ):
    dose_list = [-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1]
    defocus_list = [-40, 0, 40]

    with open(txt_path, 'w'):
        print(f'Creat {txt_path}')

    with open(txt_path, 'a') as file:
        for sub_name in os.listdir(dataset_dir):
            sub_dir = os.path.join(dataset_dir, sub_name)
            source = os.path.join(sub_dir, 'source_simple.src')
            mask = os.path.join(sub_dir, 'mask.png')
            
            if not (os.path.exists(source) and os.path.exists(mask)):
                continue

            for dose in dose_list:
                for defocus in defocus_list:
                    resist_name = f'RI_dose_{dose}_defocus_{defocus}.png'
                    resist = os.path.join(sub_dir, resist_name)

                    if os.path.exists(resist):
                        new_line = f'{source} {mask} {dose} {defocus} {resist}\n'
                        file.write(new_line)
                        print(new_line, end='')

if __name__ == "__main__":
    dataset2txt_fast()
