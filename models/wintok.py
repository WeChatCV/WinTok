import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from einops.layers.torch import Rearrange

from transformers import SiglipVisionConfig, SiglipVisionModel, Siglip2VisionConfig, Siglip2VisionModel

from models.modules import VectorQuantizerM, AttnProjection
from models.layers import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from utils.config import Args

class WinTok(nn.Module):

    def __init__(self, args):
        super().__init__()

        ### WinTok definition ###
        self.img_size = args.img_size
        self.codebook_size = args.vocab_size
        self.codebook_dim = args.vocab_width
        self.decoder_rec_depth = args.decoder_rec_depth
        self.decoder_rec_num_heads = args.decoder_rec_num_heads
        self.decoder_rec_hidden_size = args.decoder_rec_hidden_size
        self.decoder_rec_intermediate_size = args.decoder_rec_intermediate_size
        
        ## Encoder ##
        self.encoder_config = SiglipVisionConfig.from_pretrained(args.model)
        self.encoder = SiglipVisionModel.from_pretrained(args.model)
        self.encoder_hidden_dim = self.encoder_config.hidden_size

        self.patch_size = self.encoder_config.patch_size
        self.grid_size = self.encoder_config.image_size // self.patch_size
        self.num_image_tokens = self.grid_size ** 2

        self.num_latent_tokens = args.num_latent_tokens
        self.query_tokens = nn.Parameter(torch.randn(self.num_latent_tokens, self.encoder_hidden_dim))
        self.query_tokens_pos_embed = nn.Parameter(torch.randn(self.num_latent_tokens, self.encoder_hidden_dim))

        ## Decoder for reconstruction ##
        self.decoder_config = Siglip2VisionConfig()
        self.decoder_config.update({
            'patch_size': 1,
            'num_channels': self.decoder_rec_hidden_size,
            'num_hidden_layers': self.decoder_rec_depth,
            'num_attention_heads': self.decoder_rec_num_heads,
            'intermediate_size': self.decoder_rec_intermediate_size,
            'hidden_size': self.decoder_rec_hidden_size,
        })
        self.decoder = Siglip2VisionModel(self.decoder_config)

        self.decoder_rec_pos_embed = nn.Parameter(torch.randn(self.num_image_tokens, self.decoder_rec_hidden_size), requires_grad=False)

        self.decode_task_layer_1 = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, self.decoder_rec_hidden_size),
            nn.Tanh(),
            nn.Linear(self.decoder_rec_hidden_size, self.decoder_rec_hidden_size)
        )  # for projection to decoder
        self.decode_task_layer_2 = nn.Sequential(
            nn.Conv2d(self.decoder_rec_hidden_size, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)', p1 = self.patch_size, p2 = self.patch_size),
        )  # for pixel reconstruction
        self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)

        self.decode_task_layer_1.apply(self._initialize_weights)
        self.decode_task_layer_2.apply(self._initialize_weights)
        self.conv_out.apply(self._initialize_weights)

        self.scaling_layer = ScalingLayerForSigLip()

        ## quantizer ##
        self.quant_proj = AttnProjection(self.encoder_hidden_dim, self.codebook_dim, args.num_codebooks)
        self.post_quant_proj = AttnProjection(self.codebook_dim, self.encoder_hidden_dim, args.num_codebooks)
        self.quantizer = VectorQuantizerM(
            vocab_size=self.codebook_size,
            vocab_width=self.codebook_dim,
            beta=args.vq_beta,
            use_entropy_loss=args.le > 0,
            entropy_temp=args.e_temp,
            num_codebooks=args.num_codebooks,
        )
        self.quantizer.init_vocab(args.vocab_init)

        self.fc_norm = nn.LayerNorm(self.encoder_hidden_dim, eps=1e-6)
        self.projection = nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim)

        self.initialize_weights()

        ### WinTok definition ###

    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def initialize_weights(self):

        # Initialize encoder query embeddings
        query_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_dim, np.arange(self.num_latent_tokens))
        query_embed = torch.from_numpy(query_embed).float()
        self.query_tokens_pos_embed.data.copy_(query_embed)

        # Initialize decoder position embeddings
        rec_pos_embed = get_2d_sincos_pos_embed(self.decoder_rec_hidden_size, self.grid_size)
        rec_pos_embed = torch.from_numpy(rec_pos_embed).float()
        self.decoder_rec_pos_embed.data.copy_(rec_pos_embed)
        self.decoder.vision_model.embeddings = None

        # remove encoder/deocder original head
        self.encoder.vision_model.head = None
        self.decoder.vision_model.head = None

    @classmethod
    def from_checkpoint(cls, ckpt, **kwargs):
        ckpt = torch.load(ckpt, map_location='cpu', weights_only=False)
        model_cfg = Args()
        model_cfg.load_state_dict(ckpt['args'])
        model = cls(model_cfg, **kwargs)
        model.load_state_dict(ckpt['trainer']['unitok'])
        return model
        
    def encode(self, x, txt=None, **kwargs):
        x = self.scaling_layer(x)
        hidden_states = self.encoder.vision_model.embeddings(x, interpolate_pos_encoding=False)
        B, L, D = hidden_states.shape

        query_tokens = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, num_latent_tokens, D)
        query_tokens = query_tokens + self.query_tokens_pos_embed.unsqueeze(0)

        hidden_states = torch.cat([hidden_states, query_tokens], dim=1)  # (B, L+num_latent_tokens, D)

        encoder_outputs = self.encoder.vision_model.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=True,
        )

        last_hidden_state = encoder_outputs.last_hidden_state  # (B, L+num_latent_tokens, D)
        normed_hidden_state = self.encoder.vision_model.post_layernorm(last_hidden_state)

        ori_image_tokens = normed_hidden_state[:, :L, :]  # (B, L, D)

        query_tokens = normed_hidden_state[:, L:, :]  # (B, num_latent_tokens, D)

        with torch.amp.autocast("cuda", enabled=False):
            ori_image_tokens = torch.utils.checkpoint.checkpoint(self.quant_proj, ori_image_tokens.float(), use_reentrant=False)
            quant, indices, vq_loss, entropy_loss, usages = self.quantizer(ori_image_tokens)
            quant = torch.utils.checkpoint.checkpoint(self.post_quant_proj, quant, use_reentrant=False)

        query_tokens = self.projection(self.fc_norm(query_tokens))
        pooler_out = query_tokens.mean(dim=1)
        
        out_dict = {
            'pooler_out': pooler_out,
            'query_tokens': query_tokens,
            "quant": quant,
            'indices': indices,
            'vq_loss': vq_loss,
            'entropy_loss': entropy_loss,
            'usages': usages,
        }

        return out_dict

    def decode(self, quant):

        x = self.decode_task_layer_1(quant)
        B = x.shape[0]
        x = x + self.decoder_rec_pos_embed.unsqueeze(0)

        decoder_outputs = self.decoder.vision_model.encoder(
            inputs_embeds=x,
            output_hidden_states=True,
        )
        x = decoder_outputs.last_hidden_state
        x = self.decoder.vision_model.post_layernorm(x)
        x = x.permute(0, 2, 1).contiguous().view(B, -1, self.grid_size, self.grid_size)
        x = self.decode_task_layer_2(x)
        x = self.conv_out(x)

        outdict = {
            'recon': x,
        }

        return outdict

    def forward(self, img, txt=None):
        encoder_out = self.encode(img, txt=txt)
        quant = encoder_out['quant']

        decoder_out = self.decode(quant)

        outdict = {**encoder_out, **decoder_out}
        return outdict
    
    def img_to_idx(self, x):
        x = self.scaling_layer(x)
        hidden_states = self.encoder.vision_model.embeddings(x, interpolate_pos_encoding=False)
        B, L, D = hidden_states.shape

        query_tokens = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, num_latent_tokens, D)
        query_tokens = query_tokens + self.query_tokens_pos_embed.unsqueeze(0)

        hidden_states = torch.cat([hidden_states, query_tokens], dim=1)  # (B, L+num_latent_tokens, D)

        encoder_outputs = self.encoder.vision_model.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=True,
        )

        last_hidden_state = encoder_outputs.last_hidden_state  # (B, L+num_und_tokens+num_rec_tokens, D)
        normed_hidden_state = self.encoder.vision_model.post_layernorm(last_hidden_state)

        ori_image_tokens = normed_hidden_state[:, :L, :]  # (B, L, D)
        ori_image_tokens = self.quant_proj(ori_image_tokens.float())
        indices = self.quantizer.f_to_idx(ori_image_tokens)
        
        return indices

    def idx_to_img(self, indices):

        quant = self.quantizer.idx_to_f(indices)
        quant = self.post_quant_proj(quant)

        img = self.decode(quant)['recon']
        return img

    def img_to_reconstructed_img(self, image) -> torch.Tensor:
        return self.forward(image)['recon']
    
    def encode_image(self, img):
        pooler_img = self.encode(img)['pooler_out']
        pooler_img = F.normalize(pooler_img, dim=-1)
        return pooler_img

    def get_last_layer(self):
        return self.conv_out.weight

class ScalingLayerForSigLip(nn.Module):
    def __init__(self):
        super(ScalingLayerForSigLip, self).__init__()
        self.register_buffer('shift', torch.Tensor([0.5, 0.5, 0.5])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.5, 0.5, 0.5])[None, :, None, None])

    def forward(self, inp):
        inp = ((inp + 1.) * 127.5).clamp(0, 255.) / 255. # rescale to [0, 1.]
        return (inp - self.shift) / self.scale