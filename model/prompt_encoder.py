import torch.nn as nn
from torch.nn import Conv2d
from model.positional_encoding import PositionEmbeddingSine
from model.attention_layers import CrossAttentionLayer,SelfAttentionLayer,FFNLayer

class PromptEncoder(nn.Module):
    def __init__(self, backbone,in_channels=[128,256,512],num_classes=80,num_feature_level=3,hidden_dim=256,feed_forward=2048,num_layers=4,*args, **kwargs):
        super().__init__()
        self.backbone = backbone
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.PromptFeat = nn.Embedding(num_classes, hidden_dim)
        self.PromptFeatPos = nn.Embedding(num_classes, hidden_dim)
        self.cross_attn = nn.ModuleList()
        self.self_attn = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(self.num_layers):
            self.cross_attn.append(CrossAttentionLayer(hidden_dim, nhead=8))
            self.self_attn.append(SelfAttentionLayer(hidden_dim, nhead=8))
            self.ffn.append(FFNLayer(hidden_dim, feed_forward))
        self.level_embed = nn.Embedding(num_feature_level, hidden_dim)
        self.num_feature_level = num_feature_level
        self.PromptFeat.weight.data.normal_(mean=0.0, std=0.01)
        self.PromptFeatPos.weight.data.normal_(mean=0.0, std=0.01)
        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_level):
            self.input_proj.append(Conv2d(in_channels[i], hidden_dim, kernel_size=1))
        #weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x,mask=None):
        src = []
        pos = []
        size_list = []
        x = self.backbone(x)

        for i in range(self.num_feature_level):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            # flatten BxCxHxW to HWxBxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        
        bs, _, h, w = x[0].shape
        support_pos_embed = self.PromptFeatPos.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.PromptFeat.weight.unsqueeze(1).repeat(1, bs, 1)
        for layer_index in range(self.num_layers):
            level_index = layer_index % self.num_feature_level
            output = self.cross_attn[i](
                output, src[level_index],
                memory_mask=mask,
                memory_key_padding_mask=None,# here we do not apply masking on padded region
                pos=pos[level_index], query_pos=support_pos_embed
            )

            output = self.self_attn[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=support_pos_embed
            )
            
            # FFN
            output = self.ffn[i](
                output
            )
        return output

    




        