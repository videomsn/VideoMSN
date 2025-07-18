import numpy as np
import torch
import itertools

patch_drop = 0.5
patch_keep = 1. - patch_drop

new_num_patches=1764
num_frames = 9

# print(torch.arange(new_num_patches-1, dtype = torch.int8))
# import ipdb; ipdb.set_trace()

idx = torch.reshape(torch.arange(new_num_patches, dtype = torch.int),(num_frames,new_num_patches//num_frames))

# print("idx:",idx)
# exit(0)

T_H = int(np.floor((new_num_patches/num_frames)*patch_keep))
kept_patches = torch.randperm(new_num_patches//num_frames)[:T_H]

sorted_kept_patches, _ = torch.sort(kept_patches)

import ipdb; ipdb.set_trace()
mask=torch.zeros(new_num_patches, dtype=torch.bool)

mask[sorted_kept_patches] = True

slected_mask = torch.masked_select(idx, mask)

print(mask)
exit(0)


sorted_kept_patches = sorted_kept_patches.numpy()
all_kept_patches = [sorted_kept_patches + i*new_num_patches//num_frames for i in range(num_frames)]


all_kept_patches = list(itertools.chain.from_iterable(all_kept_patches))

print(T_H)

print("sorted_kept_patches: ",sorted_kept_patches)
print("all_kept_patches: ",all_kept_patches)