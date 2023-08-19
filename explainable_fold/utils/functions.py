import torch
import numpy as np
import random

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_name_dicts():

    i_to_c_dict = {0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I', 10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V', 20: 'X'}
    i_to_fc_fict = {0: '-', 1: 'A', 2: 'R', 3: 'N', 4: 'D', 5: 'C', 6: 'Q', 7: 'E', 8: 'G', 9: 'H', 10: 'I', 11: 'L', 12: 'K', 13: 'M', 14: 'F', 15: 'P', 16: 'S', 17: 'T', 18: 'W', 19: 'Y', 20: 'V', 21: 'X'}
    c_to_i_dict = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20}
    fc_to_i_dict = {'-': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, 'X': 21}
    return i_to_c_dict, i_to_fc_fict, c_to_i_dict, fc_to_i_dict


def get_cuda_info():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    return total_memory, allocated_memory


def torch_32_to_16(dict):
    for k, v in dict.items():
        if v.dtype == torch.float32:
            dict[k] = v.type(torch.bfloat16)
    return dict