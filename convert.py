import torch
from segment_anything import build_sam_vit_l

sam_checkpoint = "./models/sam_vit_l_0b3195.pth"
params = torch.load(sam_checkpoint)
new = {}

for k, v in params.items():
    v = v.numpy()
    if len(v.shape) == 2:
        if 'rel_pos_w' in k:
            pass
        elif 'rel_pos_h' in k:
            pass
        elif 'point_embeddings' in k:
            pass
        elif 'not_a_point_embed' in k:
            pass
        elif 'no_mask_embed' in k:
            pass
        elif 'iou_token' in k:
            pass
        elif 'mask_tokens' in k:
            pass
        elif 'positional_encoding_gaussian_matrix' in k:
            pass
        else:
            v = v.transpose(1, 0)
    new[k] = v

model = build_sam_vit_l()
model.set_state_dict(new)

layer_state_dict = model.state_dict()
import paddle
paddle.save(layer_state_dict, "./models/sam_vit_l_0b3195.pdparams")
