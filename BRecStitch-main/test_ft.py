import argparse
import torch
from collections import OrderedDict
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim

import cv2
#from torch_homography_model import build_model
from network import get_stitched_result_fixed, Network, build_model, build_model_ft
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import glob
from loss import cal_lp_loss3, overlap_brightness_loss, boundary_rect_loss
import torchvision.transforms as T
import grid_res

def calculate_mask_coverage_ratio(mask):

    if mask.max() > 1:
        mask = mask / 255.0
 
    valid_pixels = np.sum(mask > 0)
    
    total_pixels = mask.shape[0] * mask.shape[1]
    
    coverage_ratio = valid_pixels / total_pixels
    
    return coverage_ratio


resize_512 = T.Resize((512,512))

MODEL_DIR = '/paddle/jialing/Code/BRecStitch-main/model_new'

def test_once(input1_tensor, input2_tensor, mesh_ref, mesh_tgt, net):

    batch_size, _, img_h, img_w = input1_tensor.size()
    print("SIZE= ",img_h,img_w)
    mesh_ref = torch.stack([mesh_ref[...,0]*img_w/512, mesh_ref[...,1]*img_h/512], 3)
    mesh_tgt = torch.stack([mesh_tgt[...,0]*img_w/512, mesh_tgt[...,1]*img_h/512], 3)
    
    with torch.no_grad():
        output = get_stitched_result_fixed(input1_tensor, input2_tensor, mesh_ref, mesh_tgt)
    stitch_result = output['stitched'][0].cpu().detach().numpy().transpose(1,2,0)
    stitch_result = np.clip(stitch_result, 0, 255).astype(np.uint8)
    print(f"stitch_result: {stitch_result.shape[0]}, {stitch_result.shape[1]}")
       
    mask1 = output['mask1'][0].cpu().detach().numpy().transpose(1,2,0)
    mask2 = output['mask2'][0].cpu().detach().numpy().transpose(1,2,0)
    ave_fusion_mask = mask1 * (mask1 / (mask1 + mask2 + 1e-6)) + mask2 * (mask2 / (mask1 + mask2 + 1e-6))

    ref_image = output['warp1'][0].cpu().detach().numpy().transpose(1,2,0)
    ref_image = np.clip(ref_image, 0, 255).astype(np.uint8)
    tgt_image = output['warp2'][0].cpu().detach().numpy().transpose(1,2,0)
    tgt_image = np.clip(tgt_image, 0, 255).astype(np.uint8)
    
    mask_coverage = calculate_mask_coverage_ratio(ave_fusion_mask[:,:,0])
    
    # psnr ssim
    mask1_norm = mask1 / 255.0
    mask2_norm = mask2 / 255.0
    overlap_mask = mask1_norm * mask2_norm
    
    psnr = compare_psnr(ref_image*overlap_mask, tgt_image*overlap_mask, data_range=255)
    ssim = compare_ssim(ref_image*overlap_mask, tgt_image*overlap_mask, data_range=255, channel_axis=-1) 

    return ssim, psnr, stitch_result, mask_coverage, mask1, mask2, ave_fusion_mask, ref_image, tgt_image


def loadSingleData(img1_name, img2_name):
    # load image1
    input1 = cv2.imread(img1_name)
    input1 = cv2.resize(input1, (512, 512))
    input1 = input1.astype(dtype=np.float32)
    input1 = (input1 / 127.5) - 1.0

    # load image2
    input2 = cv2.imread(img2_name)
    input1 = cv2.resize(input1, (512, 512))
    input2 = input2.astype(dtype=np.float32)
    input2 = (input2 / 127.5) - 1.0

    max_range = 2000
    if max(input1.shape[0],input1.shape[1]) > max_range:
        scale_width = int((max_range / max(input1.shape[1],input1.shape[0])) * input1.shape[0])
        scale_hight = int((max_range / max(input1.shape[1],input1.shape[0])) * input1.shape[1])
        input1 = cv2.resize(input1,(scale_hight,scale_width), interpolation=cv2.INTER_AREA)
        input2 = cv2.resize(input2,(scale_hight,scale_width), interpolation=cv2.INTER_AREA)

    if input1.shape != input2.shape:
        input2 = cv2.resize(input2,(input1.shape[1],input1.shape[0]), interpolation=cv2.INTER_AREA)

    input1 = np.transpose(input1, [2, 0, 1])
    input2 = np.transpose(input2, [2, 0, 1])
    # convert to tensor
    input1_tensor = torch.tensor(input1).unsqueeze(0)
    input2_tensor = torch.tensor(input2).unsqueeze(0)
    return (input1_tensor, input2_tensor)

def train(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    output_dir = './finetune_metrics'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data_folder_name = os.path.basename(args.test_path.rstrip('/'))
    txt_filename = os.path.join(output_dir, f"finetune_{data_folder_name}_results.txt")

    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        print('training from stratch!')

    finetune_psnr_list = []
    finetune_ssim_list = []
    finetune_mask_list = []

    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("##################start finetune testing#######################\n")
        f.write(f"Test data path: {args.test_path}\n")
        f.write(f"Output folder: ./finetune_{data_folder_name}/\n")
        f.write("Sample_ID\tPSNR\t\tSSIM\t\tMask_Compare\tBoundary_Loss\n")
        f.write("=" * 80 + "\n")

    path_ave_other_fusion = f'./finetune_{data_folder_name}/'
    if not os.path.exists(path_ave_other_fusion):
        os.makedirs(path_ave_other_fusion)
    
    # finetune_iter_dir = './finetune_iter/'
    # if not os.path.exists(finetune_iter_dir):
    #     os.makedirs(finetune_iter_dir)

    datas = OrderedDict()
    extensions = ['*.png', '*.jpg', '*.PNG', '*.JPG', '*.jpeg', '*.JPEG']
    
    datas_list = glob.glob(os.path.join(args.test_path, '*'))

    for data in sorted(datas_list):
        data_name = data.split('/')[-1]
        if data_name == 'input-1' or data_name == 'input-2' :
            datas[data_name] = {}
            datas[data_name]['path'] = data
            full_img_list = []
            for ex in extensions:
                full_img_list.extend(glob.glob(os.path.join(data, ex)))

            datas[data_name]['image'] = full_img_list
            datas[data_name]['image'].sort()

    total_pairs = len(datas['input-1']['image'])
    print(f" {total_pairs} pairs of images")
    
    for idx in range(total_pairs):
        
        torch.cuda.empty_cache()

        # Reload the model weights
        if len(ckpt_list) != 0:
            checkpoint = torch.load(ckpt_list[-1])
            net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            scheduler.last_epoch = start_epoch
            print('finetune for a new image, init model from {}!'.format(ckpt_list[-1]))

        input1_tensor, input2_tensor = loadSingleData(img1_name = datas['input-1']['image'][idx], img2_name = datas['input-2']['image'][idx])
        if torch.cuda.is_available():
            input1_tensor = input1_tensor.cuda()
            input2_tensor = input2_tensor.cuda()
        input1_tensor_512 = resize_512(input1_tensor)
        input2_tensor_512 = resize_512(input2_tensor)
        loss_list = []

        print("##################start iteration {} #######################".format(idx+1))
        
        best_ssim, best_psnr, best_stitch_result = 0, 0, None
        best_mask1, best_mask2, best_ave_fusion_mask = None, None, None
        best_ref_image, best_tgt_image = None, None
        for epoch in range(start_epoch, start_epoch + args.max_iter):
            net.train()
            
            optimizer.zero_grad()
            batch_out = build_model_ft(net, input1_tensor_512, input2_tensor_512)
            # output_H = batch_out['output_H']
            # output_H_ref = batch_out['output_H_ref']
            # output_H_tgt = batch_out['output_H_tgt']
            output_tps_ref = batch_out['output_tps_ref']
            output_tps_tgt = batch_out['output_tps_tgt']
            mesh_ref = batch_out['mesh_ref']
            mesh_tgt = batch_out['mesh_tgt']

            # useless
            overlap_bt_loss = overlap_brightness_loss(output_tps_ref, output_tps_tgt)
            # from alignment loss
            tps_overlap_loss = cal_lp_loss3(output_tps_ref, output_tps_tgt)

            boundary_loss = boundary_rect_loss(mesh_ref, mesh_tgt)
            
            total_loss = 3.0 * tps_overlap_loss + \
                        0.001 * boundary_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()
            current_iter = epoch-start_epoch+1
            
            loss_list.append(total_loss)
            print("Training: Iteration[{:0>3}/{:0>3}] Total Loss: {:.4f} (TPS-Overlap: {:.4f}, Boundary: {:.4f}) lr={:.8f}".format(
                current_iter, args.max_iter, total_loss.item(), tps_overlap_loss.item(), boundary_loss.item(), optimizer.state_dict()['param_groups'][0]['lr']))

            # init check
            # if current_iter == 1:
            #     ssim3,psnr3,stitch_result,mask_coverage = test_once(input1_tensor, input2_tensor, mesh_ref, mesh_tgt, net)
            #     print('init psnr:{:.4f}, ssim:{:.4f}, mask:{:.4f}'.format(psnr3,ssim3,mask_coverage))
       
            #     if ssim3 > best_ssim:
            #         best_ssim = ssim3
            #         best_psnr = psnr3
            #         best_stitch_result = stitch_result

            if current_iter == args.max_iter:
                ssim3,psnr3,stitch_result,mask_coverage, mask1, mask2, ave_fusion_mask, ref_image, tgt_image = test_once(input1_tensor, input2_tensor, mesh_ref, mesh_tgt, net)
                print('final psnr:{:.4f}, ssim:{:.4f}, mask:{:.4f}'.format(psnr3,ssim3,mask_coverage))
          
                if ssim3 > best_ssim:
                    best_ssim = ssim3
                    best_psnr = psnr3
                    best_stitch_result = stitch_result
                    best_mask1 = mask1
                    best_mask2 = mask2
                    best_ave_fusion_mask = ave_fusion_mask
                    best_ref_image = ref_image
                    best_tgt_image = tgt_image

            scheduler.step()
            torch.cuda.empty_cache()

        print("##################end iteration {} #######################".format(idx+1))
        
        finetune_ssim_list.append(best_ssim)
        finetune_psnr_list.append(best_psnr)
        finetune_mask_list.append(mask_coverage)
        
        with open(txt_filename, 'a', encoding='utf-8') as f:
            f.write(f"{idx+1}\t\t{best_psnr:.6f}\t{best_ssim:.6f}\t{mask_coverage:.6f}\t\t0.000000\n")
        
        # all results
        stitched_dir = os.path.join(path_ave_other_fusion, 'stitched')
        stitch_mask_dir = os.path.join(path_ave_other_fusion, 'stitch_mask')
        warp1_dir = os.path.join(path_ave_other_fusion, 'warp1')
        warp2_dir = os.path.join(path_ave_other_fusion, 'warp2')
        mask1_dir = os.path.join(path_ave_other_fusion, 'mask1')
        mask2_dir = os.path.join(path_ave_other_fusion, 'mask2')
        fusion_mask_dir = os.path.join(path_ave_other_fusion, 'fusion_mask')
        for d in [stitched_dir, stitch_mask_dir, warp1_dir, warp2_dir, mask1_dir, mask2_dir, fusion_mask_dir]:
            os.makedirs(d, exist_ok=True)

        if best_stitch_result is not None:
            cv2.imwrite(os.path.join(stitched_dir, f"{idx+1:06d}.png"), best_stitch_result)
        if best_ave_fusion_mask is not None:
            m = np.clip(best_ave_fusion_mask, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(fusion_mask_dir, f"{idx+1:06d}.png"), m)
            cv2.imwrite(os.path.join(stitch_mask_dir, f"{idx+1:06d}.png"), m)
        if best_mask1 is not None:
            m1 = np.clip(best_mask1, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(mask1_dir, f"{idx+1:06d}.png"), m1)
        if best_mask2 is not None:
            m2 = np.clip(best_mask2, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(mask2_dir, f"{idx+1:06d}.png"), m2)
        if best_ref_image is not None:
            cv2.imwrite(os.path.join(warp1_dir, f"{idx+1:06d}.png"), best_ref_image)
        if best_tgt_image is not None:
            cv2.imwrite(os.path.join(warp2_dir, f"{idx+1:06d}.png"), best_tgt_image)

    print('average psnr:', np.mean(finetune_psnr_list))
    print('average ssim:', np.mean(finetune_ssim_list))
    print('average mask coverage:', np.mean(finetune_mask_list))
    
    with open(txt_filename, 'a', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("=================== Analysis ==================\n")
        f.write(f"Total samples: {len(finetune_psnr_list)}\n")
        f.write(f"Average PSNR: {np.mean(finetune_psnr_list):.6f}\n")
        f.write(f"Average SSIM: {np.mean(finetune_ssim_list):.6f}\n")
        f.write(f"Average Mask Compare: {np.mean(finetune_mask_list):.6f}\n")
        f.write("##################end finetune testing#######################\n")
    
    print(f"Results saved to: {txt_filename}")


if __name__=="__main__":
    print('<==================== setting arguments ===================>\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument('--test_path', type=str, default='/paddle/yun/data/test/UDIS-D/testing/')
    
    args = parser.parse_args()
    print(args)
    train(args)


