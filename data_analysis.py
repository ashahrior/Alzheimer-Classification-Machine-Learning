import os
import numpy as np

src_folder = r"E:\THESIS\ADNI_data\ADNI1_Annual_2_Yr_3T_306_WORK\INTEREST_NPY_DATA\Normalized_NPY\{}_normNPY"


def check_shape(case):
    os.chdir(src_folder.format(case))
    print(os.getcwd())
    for file in os.listdir():
        data = np.load(file, allow_pickle=True)
        print(np.max(data))
        break

if __name__ == "__main__":
    case ='MCI'
    check_shape(case)