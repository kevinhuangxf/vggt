import torch
import torch.nn as nn
import torch.nn.functional as F
from vggt.heads.dpt_head import DPTHead, custom_interpolate


class GSHead(DPTHead):
    """
    GSHead class that inherits from DPTHead.
    This class is designed for gradient-based saliency tasks and extends the functionality of DPTHead.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "relu",
        conf_activation: str = "softmax",
        features: int = 256,
        out_channels: list = [256, 512, 1024, 1024],
        intermediate_layer_idx: list = [4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
    ) -> None:
        """
        Initialize the GSHead class by calling the parent DPTHead constructor.
        """
        super().__init__(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=output_dim,
            activation=activation,
            conf_activation=conf_activation,
            features=features,
            out_channels=out_channels,
            intermediate_layer_idx=intermediate_layer_idx,
            pos_embed=pos_embed,
            feature_only=feature_only,
            down_ratio=down_ratio,
        )
        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again


    def _forward_impl(
        self, 
        aggregated_tokens_list, 
        images, 
        patch_start_idx, 
        frames_start_idx = None, 
        frames_end_idx = None
    ):
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.view(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        out = self.scratch.output_conv2(out)
        # preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        # preds = preds.view(B, S, *preds.shape[1:])
        # conf = conf.view(B, S, *conf.shape[1:])
        # return preds, conf

        num_input_views = 4
        out = out.reshape(B, num_input_views, 14, int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)) # b, 4, 14, 64, 64
        
        raw_out = out
        
        out = out.permute(0, 1, 3, 4, 2).reshape(B, -1, 14) # B, 16384, 14
        pos = self.pos_act(out[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(out[..., 3:4])
        scale = self.scale_act(out[..., 4:7])
        rotation = self.rot_act(out[..., 7:11])
        rgbs = self.rgb_act(out[..., 11:])
        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians, raw_out
