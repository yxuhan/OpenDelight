import sys
sys.path.append(".")
sys.path.append("..")

import torch
import torch.nn as nn


class DelightBaseModel(nn.Module):
    def __init__(self, enc_cfg, dec_cfg, device):
        super().__init__()
        self.device = device
        self.set_encoder(enc_cfg)
        self.set_decoder(dec_cfg)
    
    def set_encoder(self, enc_cfg):
        self.enc_type = enc_cfg["type"]
        if self.enc_type == "mae":
            from encoder.mae_encoder import get_mae_encoder
            self.encoder = get_mae_encoder(**enc_cfg, device=self.device)
        elif self.enc_type == "mae_mix":
            from encoder.mae_mix_encoder import get_mae_encoder
            self.encoder = get_mae_encoder(**enc_cfg, device=self.device)
        else:
            raise Exception

    def set_decoder(self, dec_cfg):
        self.dec_type = dec_cfg["type"]
        if self.dec_type == "sapiens":
            from decoder.sapiens_decoder import SapiensDecoder
            self.decoder = SapiensDecoder(**dec_cfg).to(self.device)
        else:
            raise Exception

    def forward(self, x, data_source=None):
        '''
        x: [b,3,h,w]  in range [0,1]
        '''
        if data_source is not None:
            return self.forward_mix(x, data_source)

        feat = self.encoder(x)
        
        if self.enc_type in ["mae"]:
            feat = feat[:, 1:, :]  # [b,hw+1,c]
            h = w = int(feat.shape[1]**.5)  # [b,hw,c]
            feat = feat.reshape(feat.shape[0], h, w, -1)  # [b,h,w,c]
            feat = feat.permute(0, 3, 1, 2).contiguous()  # [b,c,h,w]
        
        out = self.decoder(feat)
        
        out = torch.sigmoid(out)
        return out

    def forward_mix(self, x, data_source):
        assert self.enc_type == "mae_mix"
        feat = self.encoder(x, data_source)
        
        feat = feat[:, 1+self.encoder.num_reg_tokens:, :]  # [b,hw+1,c]
        h = w = int(feat.shape[1]**.5)  # [b,hw,c]
        feat = feat.reshape(feat.shape[0], h, w, -1)  # [b,h,w,c]
        feat = feat.permute(0, 3, 1, 2).contiguous()  # [b,c,h,w]

        out = self.decoder(feat)
        out = torch.sigmoid(out)
        return out


if __name__ == "__main__":
    import yaml
    with open("config/delight_base_network.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    model = DelightBaseModel(
        enc_cfg=cfg["encoder"],
        dec_cfg=cfg["decoder"],
        device="cuda"
    )
    from utils import count_parameters
    count_parameters(model, detailed=True)

    x = torch.randn(2, 3, 512, 512).cuda()
    with torch.no_grad():
        out = model(x)
    print(out.shape)  # [2,3,512,512]
