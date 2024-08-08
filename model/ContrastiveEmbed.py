import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ContrastiveEmbed(nn.Module):
    def __init__(self, max_support_len=81,norm=False):
        """
        Args:
            max_support_len: max length of support.
        """
        super().__init__()
        self.max_support_len = max_support_len
        self.norm = norm
        if norm:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.logit_scale = None

    def forward(self, x, support_dict):
        """_summary_

        Args:
            x (_type_): _description_
            support_dict (_type_): _description_
            {
                'encoded_support': encoded_support, # bs, 195, d_model
                'support_token_mask': support_token_mask, # bs, 195
                        # True for used tokens. False for padding tokens
            }
        Returns:
            _type_: _description_
        """
        assert isinstance(support_dict, dict)

        y = support_dict["encoded_support"]  #4,13,256  
        support_token_mask = support_dict["support_token_mask"]
        #print(x.shape)
        if self.norm:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
            logit_scale = self.logit_scale.exp()
            res = logit_scale * x @ y.transpose(-1, -2)
        else:
            res = x @ y.transpose(-1, -2)  
        res.masked_fill_(support_token_mask[:, None, :], float("-1e-9"))#-inf

        # padding to max_support_len
        new_res = torch.full((*res.shape[:-1], self.max_support_len), float("-1e-9"), device=res.device)#-inf
        new_res[..., : res.shape[-1]] = res

        return new_res