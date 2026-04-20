# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_model, Network, get_stitched_result_fixed, resize_512
from dataset import *
import os
import numpy as np
import skimage
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2

def calculate_mask_coverage_ratio(mask):
    if mask.max() > 1:
        mask = mask / 255.0
    
    valid_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    
    coverage_ratio = valid_pixels / total_pixels
    
    return coverage_ratio


last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model')

def get_model_dir(args):
    if args.model_path:
        return args.model_path
    else:
        return MODEL_DIR

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return


def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    base_output_dir = 'results'
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    data_folder_name = os.path.basename(args.test_path.rstrip('/'))
    # data_folder_name = "test" # Custom file name

    dataset_output_dir = os.path.join(base_output_dir, data_folder_name)
    if not os.path.exists(dataset_output_dir):
        os.makedirs(dataset_output_dir)
    
    test_results_dir = os.path.join(dataset_output_dir, 'test-results')
    test_results_mask_dir = os.path.join(dataset_output_dir, 'test-results-mask')
    test_warped_1_dir = os.path.join(dataset_output_dir, 'test-warped-1')
    test_warped_2_dir = os.path.join(dataset_output_dir, 'test-warped-2')
    test_warped_mask_1_dir = os.path.join(dataset_output_dir, 'test-warped-mask-1')
    test_warped_mask_2_dir = os.path.join(dataset_output_dir, 'test-warped-mask-2')
    
    for dir_path in [test_results_dir, test_results_mask_dir, test_warped_1_dir, 
                    test_warped_2_dir, test_warped_mask_1_dir, test_warped_mask_2_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    txt_filename = os.path.join(dataset_output_dir, f'{data_folder_name}_results.txt')
    
    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=False)

    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    #load the existing models if it exists
    model_dir = get_model_dir(args)
    ckpt_list = glob.glob(model_dir + "/*.pth")
    
    print('model_dir = ', model_dir)
    
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')


    print("##################start testing#######################")
    psnr_list = []
    ssim_list = []
    mask_list = []  
    net.eval()
    
    # writing results
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("##################start testing#######################\n")
        f.write(f"Test data path: {args.test_path}\n")
        f.write(f"Model path: {model_path if 'model_path' in locals() else 'No model loaded'}\n")
        f.write("=" * 50 + "\n")
    
    for i, batch_value in enumerate(test_loader):

        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()

        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

            input1_tensor = resize_512(inpu1_tesnor)
            input2_tensor = resize_512(inpu2_tesnor)

            with torch.no_grad():
                batch_out = build_model(net, input1_tensor, input2_tensor, is_training=False)

            # output_tps = batch_out['output_H']
            # output_tps_ref = batch_out['output_tps_ref']
            # output_tps_tgt = batch_out['output_tps_tgt']
            mesh_ref = batch_out['mesh_ref']
            mesh_tgt = batch_out['mesh_tgt']
            
            batch_size, _, img_h, img_w = inpu1_tesnor.size()
            mesh_ref = torch.stack([mesh_ref[...,0]*img_w/512, mesh_ref[...,1]*img_h/512], 3)
            mesh_tgt = torch.stack([mesh_tgt[...,0]*img_w/512, mesh_tgt[...,1]*img_h/512], 3)
            
            with torch.no_grad():
                output = get_stitched_result_fixed(inpu1_tesnor, inpu2_tesnor, mesh_ref, mesh_tgt)
            
            mask1 = output['mask1'][0].cpu().detach().numpy().transpose(1,2,0)
            mask2 = output['mask2'][0].cpu().detach().numpy().transpose(1,2,0)

            stitched_image = output['stitched'][0].cpu().detach().numpy().transpose(1,2,0)
            stitched_image = np.clip(stitched_image, 0, 255).astype(np.uint8)
            
            # Replace it with a white background
            # valid_mask = np.any(stitched_image > 0, axis=2, keepdims=True)
            # stitched_image = np.where(valid_mask, stitched_image, 255)
    
            stitched_image_path = os.path.join(test_results_dir, f"{i+1:06d}.png")
            cv2.imwrite(stitched_image_path, stitched_image)
            
            ave_fusion_mask = mask1 * (mask1 / (mask1 + mask2 + 1e-6)) + mask2 * (mask2 / (mask1 + mask2 + 1e-6))
            
            mask_coverage = calculate_mask_coverage_ratio(ave_fusion_mask[:,:,0])

            mask_image = np.clip(ave_fusion_mask[:,:,0] * 255, 0, 255).astype(np.uint8)
            mask_image_path = os.path.join(test_results_mask_dir, f"{i+1:06d}.png")
            cv2.imwrite(mask_image_path, mask_image)
            
            ref_image = output['warp1'][0].cpu().detach().numpy().transpose(1,2,0)
            ref_image = np.clip(ref_image, 0, 255).astype(np.uint8)
            ref_mask = mask1[:,:,0]
            ref_mask = np.clip(ref_mask, 0, 255).astype(np.uint8)
            
            ref_image_path = os.path.join(test_warped_1_dir, f"{i+1:06d}.png")
            ref_mask_path = os.path.join(test_warped_mask_1_dir, f"{i+1:06d}.png")
            cv2.imwrite(ref_image_path, ref_image)
            cv2.imwrite(ref_mask_path, ref_mask)
            
            tgt_image = output['warp2'][0].cpu().detach().numpy().transpose(1,2,0)
            tgt_image = np.clip(tgt_image, 0, 255).astype(np.uint8)
            tgt_mask = mask2[:,:,0]
            tgt_mask = np.clip(tgt_mask, 0, 255).astype(np.uint8)
            
            tgt_image_path = os.path.join(test_warped_2_dir, f"{i+1:06d}.png")
            tgt_mask_path = os.path.join(test_warped_mask_2_dir, f"{i+1:06d}.png")
            cv2.imwrite(tgt_image_path, tgt_image)
            cv2.imwrite(tgt_mask_path, tgt_mask)
            
            # psnr ssim
            mask1_norm = mask1 / 255.0
            mask2_norm = mask2 / 255.0
            overlap_mask = mask1_norm * mask2_norm
            
            psnr = compare_psnr(ref_image*overlap_mask, tgt_image*overlap_mask, data_range=255)
            ssim = compare_ssim(ref_image*overlap_mask, tgt_image*overlap_mask, data_range=255, channel_axis=-1)        

            print('i = {}, psnr = {:.4f}, ssim = {:.4f}, mask = {:.4f}'.format( i+1, psnr, ssim, mask_coverage))

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            mask_list.append(mask_coverage)
            
            # Write results to txt file
            with open(txt_filename, 'a', encoding='utf-8') as f:
                f.write('i = {}, psnr = {:.4f}, ssim = {:.4f}, mask = {:.4f}\n'.format( i+1, psnr, ssim, mask_coverage))
                       
            torch.cuda.empty_cache()

    print("=================== Analysis ==================")
    print("psnr")
    psnr_list.sort(reverse = True)
    print('average psnr:', np.mean(psnr_list))

    ssim_list.sort(reverse = True)
    print('average ssim:', np.mean(ssim_list))
    
    print("mask coverage")
    mask_list.sort(reverse = True)
    print('average mask coverage:', np.mean(mask_list))
    print("##################end testing#######################")
    
    with open(txt_filename, 'a', encoding='utf-8') as f:
        f.write("\n=================== Analysis ==================\n")
        f.write('average psnr: {}\n'.format(np.mean(psnr_list)))
        f.write('average ssim: {}\n'.format(np.mean(ssim_list)))
        f.write('average mask coverage: {}\n'.format(np.mean(mask_list)))
        f.write("##################end testing#######################\n")
    
    print(f"Results saved to: {txt_filename}")


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='/paddle/yun/data/test/UDIS-D/testing/')

    parser.add_argument('--model_path', type=str, default='/paddle/jialing/Code/BRecStitch-main/model_new')
    
    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
