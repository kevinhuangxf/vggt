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
from vggt.models.vggt_cam_gs import VGGT_CAM_GS
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Initialize the SSIM metric
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

minmax_norm = lambda x: (x - x.min()) / (x.max() - x.min())

class GaussianRendererWrapper:
    def __init__(self, opt, model):

        self.gs = GaussianRenderer(opt)
        self.opt = opt
        self.model = model

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg').cuda()
            self.lpips_loss.requires_grad_(False)

    def render_gs(self, data):
        results = {}
        loss = 0

        # use the first view to predict gaussians
        
        images = data['input'][:, :, :3, ...] # [B, 4, 9, h, W], input features
        B, V, C, H, W = images.shape
        # images = images.view(B*V, C, H, W) # [B*V, 9, h, W], flatten the batch and view dimensions
        # gaussians = self.forward_gaussians(images) # [B, N, 14]
        # images = images[None]  # Add batch dimension
        aggregated_tokens_list, ps_idx = self.model.aggregator(images)
        gaussians = self.model.gs_head(aggregated_tokens_list, images, ps_idx)

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
                    
                    if opt.use_depth:
                        gt_depths = data['depths_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_depths = gt_depths.transpose(0, 3, 1, 4, 2).reshape(-1, gt_depths.shape[1] * gt_depths.shape[3], 1) # [B*output_size, V*output_size, 1]
                        gt_depths = np.repeat(gt_depths, repeats=3, axis=-1) # [B*output_size, V*output_size, 3]

                        pred_depths = out['depths_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                        pred_depths = pred_depths.transpose(0, 3, 1, 4, 2)
                        pred_depths = pred_depths.reshape(opt.output_size * opt.batch_size, opt.output_size * opt.num_views, 1)
                        pred_depths = np.repeat(pred_depths, repeats=3, axis=-1)

                        # depths_mast3r = out['depths_mast3r'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                        # depths_mast3r = depths_mast3r.transpose(0, 3, 1, 4, 2)
                        # depths_mast3r = depths_mast3r.reshape(opt.output_size * opt.batch_size, opt.output_size * opt.num_input_views, 1)
                        # depths_mast3r = np.repeat(depths_mast3r, repeats=3, axis=-1)

                        # pred_depths = np.concatenate((pred_depths, pred_depths), axis=1)
                        # pred_depths = np.concatenate((pred_depths, depths_mast3r), axis=1)
                        
                        # print(images_combined.shape, gt_depths.shape, pred_depths.shape)
                        images_combined = np.concatenate((images_combined, gt_depths, pred_depths), axis=0)

                        depth_mast3r = out['depths_mast3r'].detach().cpu().numpy()
                        depth_mast3r = depth_mast3r.transpose(1, 0, 2)
                        depth_mast3r = depth_mast3r.reshape(opt.output_size * opt.batch_size, opt.output_size * opt.num_input_views, 1)
                        depth_mast3r = np.repeat(depth_mast3r, repeats=3, axis=-1)
                        depth_mast3r = np.concatenate((depth_mast3r, depth_mast3r), axis=1)
                        images_combined = np.concatenate((images_combined, depth_mast3r), axis=0)
                    
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

        # eval
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            total_ssim = 0
            total_lpips = 0
            for i, data in enumerate(test_dataloader):

                out = model(data)
    
                psnr = out['psnr']
                total_psnr += psnr.detach()
                # ssim = out['ssim']
                # total_ssim += ssim.detach()
                lpips = out['lpips']
                total_lpips += lpips.detach()
                
                # save some images
                if accelerator.is_main_process:
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)

                    # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/eval_pred_alphas_{epoch}_{i}.jpg', pred_alphas)

                    # Concatenate pred_images and gt_images vertically
                    images_combined = np.concatenate((gt_images, pred_images), axis=0)

                    if opt.use_depth:
                        gt_depths = data['depths_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_depths = gt_depths.transpose(0, 3, 1, 4, 2).reshape(-1, gt_depths.shape[1] * gt_depths.shape[3], 1) # [B*output_size, V*output_size, 1]
                        gt_depths = np.repeat(gt_depths, repeats=3, axis=-1) # [B*output_size, V*output_size, 3]

                        gt_masks = data['masks_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_masks = gt_masks.transpose(0, 3, 1, 4, 2).reshape(-1, gt_masks.shape[1] * gt_masks.shape[3], 1) # [B*output_size, V*output_size, 1]
                        gt_masks = np.repeat(gt_masks, repeats=3, axis=-1) # [B*output_size, V*output_size, 3]

                        pred_depths = out['depths_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                        pred_depths = pred_depths.transpose(0, 3, 1, 4, 2).reshape(-1, pred_depths.shape[1] * pred_depths.shape[3], 1) # [B*output_size, V*output_size, 1]
                        # pred_depths = pred_depths.transpose(0, 3, 1, 4, 2)
                        # pred_depths = pred_depths.reshape(opt.output_size * opt.batch_size, opt.output_size * opt.num_input_views, 1)
                        pred_depths = np.repeat(pred_depths, repeats=3, axis=-1)

                        # print(gt_depths.shape, pred_depths.shape)
                        # depth_errs = compute_errors(gt_depths, pred_depths, gt_masks)
                        # print("gt pred depth_errs: ", depth_errs)

                        # depths_mast3r = out['depths_mast3r'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                        # depths_mast3r = depths_mast3r.transpose(0, 3, 1, 4, 2)
                        # depths_mast3r = np.concatenate((depths_mast3r[:, :, ::2, ...], depths_mast3r[:, :, 1::2, ...]), axis=2)
                        # depths_mast3r = depths_mast3r.reshape(opt.output_size * opt.batch_size, opt.output_size * opt.num_input_views, 1)
                        # depths_mast3r = np.repeat(depths_mast3r, repeats=3, axis=-1)
                        # depths_mast3r_zeropad = np.zeros_like(pred_depths)
                        # depths_mast3r_zeropad[:, :depths_mast3r.shape[1]] = depths_mast3r

                        # depth_errs = compute_errors(gt_depths[0], pred_depths[0])
                        # print("gt pred depth_errs: ", depth_errs)

                        # pred_depths = np.concatenate((depths_mast3r, pred_depths), axis=1)
                        # pred_depths = np.concatenate((pred_depths, depths_mast3r), axis=1)
                        
                        # images_combined = np.concatenate((images_combined, gt_depths, pred_depths, depths_mast3r_zeropad), axis=0)
                        images_combined = np.concatenate((images_combined, gt_depths, pred_depths), axis=0)
                        kiui.write_image(f'{opt.workspace}/images_combined_{epoch}_{i}.jpg', images_combined)                                        

            torch.cuda.empty_cache()

            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            if accelerator.is_main_process:
                total_psnr /= len(test_dataloader)
                accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")
                # total_ssim /= len(test_dataloader)
                # accelerator.print(f"[eval] epoch: {epoch} ssim: {ssim:.4f}")
                total_lpips /= len(test_dataloader)
                accelerator.print(f"[eval] epoch: {epoch} lpips: {lpips:.4f}")



if __name__ == "__main__":
    main()
