import kiui
import tyro
import numpy as np

import torch
import torch.nn.functional as F
from core.options import AllConfigs
from core.gs import GaussianRenderer
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator, DistributedDataParallelKwargs

from kiui.lpips import LPIPS
from vggt.models.vggt import VGGT
from vggt.models.vggt_cam_gs import VGGT_CAM_GS
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Initialize the SSIM metric
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

import trimesh

import os
# # 设置 master_port
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29500'

minmax_norm = lambda x: (x - x.min()) / (x.max() - x.min())

class GaussianRendererWrapper:
    def __init__(self, opt, model):

        self.gs = GaussianRenderer(opt)
        self.opt = opt
        self.model = model
        # self.vggt = VGGT.from_pretrained("facebook/VGGT-1B").cuda()

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg').cuda()
            self.lpips_loss.requires_grad_(False)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

    def depthmap_to_absolute_camera_coordinates(self, depthmaps, camera_intrinsics, camera_poses, **kw):
        """
        Args:
            - depthmaps (BxHxW tensor): Batch of depth maps
            - camera_intrinsics: a 3x3 matrix
            - camera_poses: a batch of 4x3 or 4x4 cam2world matrices (Bx4x4 or Bx4x3)
        Returns:
            pointmap of absolute coordinates (BxHxWx3 tensor), and a mask specifying valid pixels.
        """

        def depthmap_to_camera_coordinates(depthmaps, camera_intrinsics, pseudo_focal=None):
            """
            Args:
                - depthmaps (BxHxW tensor): Batch of depth maps
                - camera_intrinsics: a 3x3 matrix
            Returns:
                pointmap of absolute coordinates (BxHxWx3 tensor), and a mask specifying valid pixels.
            """
            camera_intrinsics = torch.tensor(camera_intrinsics, dtype=torch.float32)
            B, H, W = depthmaps.shape

            # Compute 3D ray associated with each pixel
            # Strong assumption: there are no skew terms
            assert camera_intrinsics[0, 1] == 0.0
            assert camera_intrinsics[1, 0] == 0.0
            if pseudo_focal is None:
                fu = camera_intrinsics[0, 0]
                fv = camera_intrinsics[1, 1]
            else:
                assert pseudo_focal.shape == (H, W)
                fu = fv = pseudo_focal
            cu = camera_intrinsics[0, 2]
            cv = camera_intrinsics[1, 2]
            device = camera_intrinsics.device

            u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
            u = u.unsqueeze(0).expand(B, -1, -1)
            v = v.unsqueeze(0).expand(B, -1, -1)
            z_cam = depthmaps
            x_cam = (u - cu) * z_cam / fu
            y_cam = (v - cv) * z_cam / fv
            X_cam = torch.stack((x_cam, y_cam, z_cam), dim=-1).to(torch.float32)

            # Mask for valid coordinates
            valid_mask = (depthmaps > 0.0)
            return X_cam, valid_mask

        X_cam, valid_mask = depthmap_to_camera_coordinates(depthmaps, camera_intrinsics)

        B, H, W, _ = X_cam.shape
        R_cam2world = camera_poses[:, :3, :3]
        t_cam2world = camera_poses[:, :3, 3]

        # print("R_cam2world: ", R_cam2world.shape)
        # print("X_cam: ", X_cam.shape)

        # Express in absolute coordinates (invalid depth values)
        X_world = torch.einsum("bij,bhwj->bhwi", R_cam2world, X_cam) + t_cam2world[:, None, None, :]
        return X_world

    def normalize_point_cloud(self, point_clouds, box_scale):
        """
        Args:
            - point_clouds (BxNx3 tensor): Batch of point clouds
            - box_scale (float): Scale factor for normalization
        Returns:
            - normalized_point_clouds (BxNx3 tensor): Normalized point clouds
        """
        # Compute the bounding box for each point cloud
        bbox_min = torch.min(point_clouds, dim=1, keepdim=True).values
        bbox_max = torch.max(point_clouds, dim=1, keepdim=True).values

        # Compute the scale factor
        scale = box_scale / torch.max(bbox_max - bbox_min, dim=2, keepdim=True).values

        # Scale the point clouds
        point_clouds = point_clouds * scale

        # Recompute the bounding box
        bbox_min = torch.min(point_clouds, dim=1, keepdim=True).values
        bbox_max = torch.max(point_clouds, dim=1, keepdim=True).values

        # Compute the offset
        offset = -(bbox_min + bbox_max) / 2

        # Translate the point clouds
        normalized_point_clouds = point_clouds + offset

        return normalized_point_clouds

    def render_gs(self, data):
        results = {}
        loss = 0

        # use the first view to predict gaussians
        
        images = data['input'][:, :, :3, ...] # [B, 4, 9, h, W], input features
        B, V, C, H, W = images.shape

        masks_input = data['masks_output'][:, :4, ...]
        masks_input = (masks_input > 0.5).float().permute(0, 1, 3, 4, 2).reshape(B, -1, 1)

        depth_input = data['depths_output'][:, :4, ...]

        # images = images.view(B*V, C, H, W) # [B*V, 9, h, W], flatten the batch and view dimensions
        # gaussians = self.forward_gaussians(images) # [B, N, 14]
        # images = images[None]  # Add batch dimension
        aggregated_tokens_list, ps_idx = self.model.module.aggregator(images)
        gaussians, raw_out = self.model.module.gs_head(aggregated_tokens_list, images, ps_idx)
        # gs_out = gs_out.reshape(B, self.opt.num_input_views, 14, H, W) # b, 4, 14, 64, 64
        # gs_out = gs_out.permute(0, 1, 3, 4, 2).reshape(B, -1, 14) # B, 16384, 14

        # pose = data['extrinsics'][:, :self.opt.num_input_views]
        # pose = pose.reshape(B*V, 4, 4)
        # # depth_map, depth_conf = self.model.module.depth_head(aggregated_tokens_list, images, ps_idx)
        # pts = self.depthmap_to_absolute_camera_coordinates(depth_input.view(B*V, H, W, 1)[..., 0], data['intrinsics'][0][0], pose)
        # pts = self.normalize_point_cloud(pts.reshape(B*V, -1, 3), 2.0).reshape(B, -1, 3)

        # point_map, point_conf  = self.vggt.point_head(aggregated_tokens_list, images, ps_idx)
        # pts = self.normalize_point_cloud(point_map.reshape(B, -1, 3), 2.0).reshape(B, -1, 3)

        # gs_out[:, :, :3] = pts # gs_out[:, :, :3] * 0.5
        # gs_out[:, :, 3:4] = masks_input
        # gs_out[:, :, 11:] = images.permute(0, 1, 3, 4, 2).reshape(B, -1, 3)

        # pos = self.pos_act(gs_out[..., 0:3]) # [B, N, 3]
        # opacity = self.opacity_act(gs_out[..., 3:4])
        # scale = self.scale_act(gs_out[..., 4:7])
        # rotation = self.rot_act(gs_out[..., 7:11])
        # rgbs = self.rgb_act(gs_out[..., 11:])
        # gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]


        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) 
        gs_results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color) # [B, V, C, output_size, output_size]
        pred_images = gs_results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = gs_results['alpha'] # [B, V, 1, output_size, output_size]
        pred_depths = gs_results['depth'] # [B, V, 1, output_size, output_size]

        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        pred_depths = minmax_norm(pred_depths)
        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['depths_pred'] = pred_depths

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            ssim_score = ssim(pred_images.reshape(-1, 3, self.opt.output_size, self.opt.output_size).cpu(), gt_images.reshape(-1, 3, self.opt.output_size, self.opt.output_size).cpu())
            # psnr_score = psnr_metric(pred_images.detach(), gt_images)

            results['lpips'] = loss_lpips
            results['ssim'] = ssim_score.item()
            results['psnr'] = psnr
            results['gaussians'] = gaussians
        
        return results

def main():    
    opt = tyro.cli(AllConfigs)

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    # Initialize the VGGT_CAM_GS model and load pretrained weights
    model = VGGT_CAM_GS.from_pretrained("facebook/VGGT-1B").to(device)
    
    # data
    if opt.data_mode == 's3':
        from core.provider_objaverse import ObjaverseDataset as Dataset
    elif opt.data_mode == 'dust3r':
        from core.provider_objaverse_dust3r import ObjaverseDataset as Dataset
    elif opt.data_mode == 'shapesplat':
        from core.provider_objaverse_dust3r import ShapeSplatDataset as Dataset
    else:
        raise NotImplementedError

    train_dataset = Dataset(opt, training=True)
    print("train_dataset: ", len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.gs_head.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    # scheduler (per-iteration)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 3000 / total_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    gs_wrapper = GaussianRendererWrapper(opt, model)

    # loop
    for epoch in range(opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                # out = model(data, step_ratio)
                out = gs_wrapper.render_gs(data)

                loss = out['loss']
                psnr = out['psnr']
                accelerator.backward(loss)

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()

            if accelerator.is_main_process:
                # logging
                if i % opt.log_steps == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}")
                
                # save log images
                # if i % 100 == 0:
                if i % opt.log_steps == 0:
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    # kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images)
                    # if not os.path.exists(f'{opt.workspace}/train_gt.jpg'):
                    #     kiui.write_image(f'{opt.workspace}/train_gt.jpg', gt_images)

                    # gt_alphas = data['masks_output'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # gt_alphas = gt_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, gt_alphas.shape[1] * gt_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/train_gt_alphas_{epoch}_{i}.jpg', gt_alphas)
                    
                    # Concatenate pred_images and gt_images vertically
                    images_combined = np.concatenate((gt_images, pred_images), axis=0)
                    
                    # if opt.use_depth:
                    #     gt_depths = data['depths_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    #     gt_depths = gt_depths.transpose(0, 3, 1, 4, 2).reshape(-1, gt_depths.shape[1] * gt_depths.shape[3], 1) # [B*output_size, V*output_size, 1]
                    #     gt_depths = np.repeat(gt_depths, repeats=3, axis=-1) # [B*output_size, V*output_size, 3]

                    #     pred_depths = out['depths_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    #     pred_depths = pred_depths.transpose(0, 3, 1, 4, 2)
                    #     pred_depths = pred_depths.reshape(opt.output_size * opt.batch_size, opt.output_size * opt.num_views, 1)
                    #     pred_depths = np.repeat(pred_depths, repeats=3, axis=-1)

                    #     # depths_mast3r = out['depths_mast3r'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    #     # depths_mast3r = depths_mast3r.transpose(0, 3, 1, 4, 2)
                    #     # depths_mast3r = depths_mast3r.reshape(opt.output_size * opt.batch_size, opt.output_size * opt.num_input_views, 1)
                    #     # depths_mast3r = np.repeat(depths_mast3r, repeats=3, axis=-1)

                    #     # pred_depths = np.concatenate((pred_depths, pred_depths), axis=1)
                    #     # pred_depths = np.concatenate((pred_depths, depths_mast3r), axis=1)
                        
                    #     # print(images_combined.shape, gt_depths.shape, pred_depths.shape)
                    #     images_combined = np.concatenate((images_combined, gt_depths, pred_depths), axis=0)

                    #     depth_mast3r = out['depths_mast3r'].detach().cpu().numpy()
                    #     depth_mast3r = depth_mast3r.transpose(1, 0, 2)
                    #     depth_mast3r = depth_mast3r.reshape(opt.output_size * opt.batch_size, opt.output_size * opt.num_input_views, 1)
                    #     depth_mast3r = np.repeat(depth_mast3r, repeats=3, axis=-1)
                    #     depth_mast3r = np.concatenate((depth_mast3r, depth_mast3r), axis=1)
                    #     images_combined = np.concatenate((images_combined, depth_mast3r), axis=0)
                    
                    kiui.write_image(f'{opt.workspace}/train_pred_gt_images_{epoch}_{i}.jpg', images_combined)
                    # kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}_{i}.jpg', pred_images)

                    # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/train_pred_alphas_{epoch}_{i}.jpg', pred_alphas)

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
        
        # checkpoint
        # if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
        accelerator.wait_for_everyone()
        # TODO: save model at different epochs
        accelerator.save_model(model, opt.workspace)

        torch.cuda.empty_cache()

        # # eval
        # with torch.no_grad():
        #     model.eval()
        #     total_psnr = 0
        #     total_ssim = 0
        #     total_lpips = 0
        #     for i, data in enumerate(test_dataloader):

        #         out = model(data)
    
        #         psnr = out['psnr']
        #         total_psnr += psnr.detach()
        #         # ssim = out['ssim']
        #         # total_ssim += ssim.detach()
        #         lpips = out['lpips']
        #         total_lpips += lpips.detach()
                
        #         # save some images
        #         if accelerator.is_main_process:
        #             gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
        #             gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
        #             kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

        #             pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
        #             pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
        #             kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)

        #             # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
        #             # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
        #             # kiui.write_image(f'{opt.workspace}/eval_pred_alphas_{epoch}_{i}.jpg', pred_alphas)

        #             # Concatenate pred_images and gt_images vertically
        #             images_combined = np.concatenate((gt_images, pred_images), axis=0)

        #             if opt.use_depth:
        #                 gt_depths = data['depths_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
        #                 gt_depths = gt_depths.transpose(0, 3, 1, 4, 2).reshape(-1, gt_depths.shape[1] * gt_depths.shape[3], 1) # [B*output_size, V*output_size, 1]
        #                 gt_depths = np.repeat(gt_depths, repeats=3, axis=-1) # [B*output_size, V*output_size, 3]

        #                 gt_masks = data['masks_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
        #                 gt_masks = gt_masks.transpose(0, 3, 1, 4, 2).reshape(-1, gt_masks.shape[1] * gt_masks.shape[3], 1) # [B*output_size, V*output_size, 1]
        #                 gt_masks = np.repeat(gt_masks, repeats=3, axis=-1) # [B*output_size, V*output_size, 3]

        #                 pred_depths = out['depths_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
        #                 pred_depths = pred_depths.transpose(0, 3, 1, 4, 2).reshape(-1, pred_depths.shape[1] * pred_depths.shape[3], 1) # [B*output_size, V*output_size, 1]
        #                 # pred_depths = pred_depths.transpose(0, 3, 1, 4, 2)
        #                 # pred_depths = pred_depths.reshape(opt.output_size * opt.batch_size, opt.output_size * opt.num_input_views, 1)
        #                 pred_depths = np.repeat(pred_depths, repeats=3, axis=-1)

        #                 # print(gt_depths.shape, pred_depths.shape)
        #                 # depth_errs = compute_errors(gt_depths, pred_depths, gt_masks)
        #                 # print("gt pred depth_errs: ", depth_errs)

        #                 # depths_mast3r = out['depths_mast3r'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
        #                 # depths_mast3r = depths_mast3r.transpose(0, 3, 1, 4, 2)
        #                 # depths_mast3r = np.concatenate((depths_mast3r[:, :, ::2, ...], depths_mast3r[:, :, 1::2, ...]), axis=2)
        #                 # depths_mast3r = depths_mast3r.reshape(opt.output_size * opt.batch_size, opt.output_size * opt.num_input_views, 1)
        #                 # depths_mast3r = np.repeat(depths_mast3r, repeats=3, axis=-1)
        #                 # depths_mast3r_zeropad = np.zeros_like(pred_depths)
        #                 # depths_mast3r_zeropad[:, :depths_mast3r.shape[1]] = depths_mast3r

        #                 # depth_errs = compute_errors(gt_depths[0], pred_depths[0])
        #                 # print("gt pred depth_errs: ", depth_errs)

        #                 # pred_depths = np.concatenate((depths_mast3r, pred_depths), axis=1)
        #                 # pred_depths = np.concatenate((pred_depths, depths_mast3r), axis=1)
                        
        #                 # images_combined = np.concatenate((images_combined, gt_depths, pred_depths, depths_mast3r_zeropad), axis=0)
        #                 images_combined = np.concatenate((images_combined, gt_depths, pred_depths), axis=0)
        #                 kiui.write_image(f'{opt.workspace}/images_combined_{epoch}_{i}.jpg', images_combined)                                        

        #     torch.cuda.empty_cache()

        #     total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        #     if accelerator.is_main_process:
        #         total_psnr /= len(test_dataloader)
        #         accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")
        #         # total_ssim /= len(test_dataloader)
        #         # accelerator.print(f"[eval] epoch: {epoch} ssim: {ssim:.4f}")
        #         total_lpips /= len(test_dataloader)
        #         accelerator.print(f"[eval] epoch: {epoch} lpips: {lpips:.4f}")



if __name__ == "__main__":
    main()
