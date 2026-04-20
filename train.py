import argparse
import skimage.measure
import torch
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import build_model, Network
from dataset import TrainDataset, TestDataset
import glob
from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss, boundary_rect_loss

import skimage
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np

last_path = os.path.abspath(os.path.dirname("__file__"))
# path to save the summary files
SUMMARY_DIR = os.path.join(last_path, 'summary')
writer = SummaryWriter(log_dir=SUMMARY_DIR)
# path to save the model files
MODEL_DIR = os.path.join(last_path, 'model')

# create folders if it dose not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

def train(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_data = TrainDataset(
        data_path=args.train_path,
    )
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=True)

    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()
    
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)

        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('training from scratch!')

    print("##################start training#######################")
    score_print_fre = 1000 

    for epoch in range(start_epoch, args.max_epoch):
        
        print("start epoch {}".format(epoch))
        net.train() 
        loss_sigma = 0.0
        overlap_loss_sigma = 0.
        midplane_loss_sigma = 0.
        nonoverlap_loss_sigma = 0.
        boundary_loss_sigma = 0.
        
        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, batch_value in enumerate(train_loader): 

            input1_tesnor = batch_value[0].float()
            input2_tesnor = batch_value[1].float()

            if torch.cuda.is_available():
                input1_tesnor = input1_tesnor.cuda()
                input2_tesnor = input2_tesnor.cuda()

            optimizer.zero_grad() 

            batch_out = build_model(net, input1_tesnor, input2_tesnor)
            # result: homo
            output_H = batch_out['output_H']
            output_H_ref = batch_out['output_H_ref']
            output_H_tgt = batch_out['output_H_tgt']
            # result: tps
            output_tps_ref = batch_out['output_tps_ref']
            output_tps_tgt = batch_out['output_tps_tgt']
            mesh_ref = batch_out['mesh_ref']
            mesh_tgt = batch_out['mesh_tgt']
            

            # alignment loss
            overlap_loss = cal_lp_loss(input1_tesnor, output_H, output_H_ref, output_H_tgt, output_tps_ref, output_tps_tgt)

            # useless
            midplane_loss =  overlap_loss * 0 

            # shape loss
            nonoverlap_loss_ref = 10*inter_grid_loss(mesh_ref) + 10*intra_grid_loss(mesh_ref) 
            nonoverlap_loss_tgt = 10*inter_grid_loss(mesh_tgt) + 10*intra_grid_loss(mesh_tgt) 
            nonoverlap_loss = nonoverlap_loss_ref + nonoverlap_loss_tgt 
            
            # boundary loss
            boundary_loss = boundary_rect_loss(mesh_ref, mesh_tgt)
            
            total_loss = ( overlap_loss + 
                        midplane_loss + 
                        nonoverlap_loss + 
                        0.0001 * boundary_loss)      
            
            total_loss.backward() # 反向传播

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2) 
            optimizer.step() 

            overlap_loss_sigma += overlap_loss.item() 
            midplane_loss_sigma += midplane_loss.item() 
            nonoverlap_loss_sigma += nonoverlap_loss.item() 
            boundary_loss_sigma += boundary_loss.item() 
            loss_sigma += total_loss.item() 

            print(glob_iter)

            if i % score_print_fre == 0 and i != 0:
                average_loss = loss_sigma / score_print_fre
                average_overlap_loss = overlap_loss_sigma/ score_print_fre
                average_midplane_loss = midplane_loss_sigma/ score_print_fre
                average_nonoverlap_loss = nonoverlap_loss_sigma/ score_print_fre
                average_boundary_loss = boundary_loss_sigma/ score_print_fre
                
                loss_sigma = 0.0
                overlap_loss_sigma = 0.
                midplane_loss_sigma = 0.
                nonoverlap_loss_sigma = 0.
                boundary_loss_sigma = 0.

                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Overlap Loss: {:.4f} Mid-plane Loss: {:.4f} Non-overlap Loss: {:.4f} Boundary Loss: {:.4f} lr={:.8f}".format(
                    epoch + 1, args.max_epoch, i + 1, len(train_loader), average_loss, average_overlap_loss, 
                    average_midplane_loss, average_nonoverlap_loss, average_boundary_loss, 
                    optimizer.state_dict()['param_groups'][0]['lr']))
                
                # visualization
                writer.add_image("input1", (input1_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("input2", (input2_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("output_2H", ((output_H_ref[0,0:3,:,:] + output_H_tgt[0,0:3,:,:])/2 + 1.)/2., glob_iter)
                writer.add_image("output_2Mesh", ((output_tps_ref[0,0:3,:,:] + output_tps_tgt[0,0:3,:,:])/2 + 1.)/2., glob_iter)

                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_loss, glob_iter)
                writer.add_scalar('overlap loss', average_overlap_loss, glob_iter)
                writer.add_scalar('mdiplane loss', average_midplane_loss, glob_iter)
                writer.add_scalar('nonoverlap loss', average_nonoverlap_loss, glob_iter)
                writer.add_scalar('boundary loss', average_boundary_loss, glob_iter)


            glob_iter += 1

        scheduler.step()

        # save model
        if ((epoch+1) % 10 == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)


        # testing
        if (epoch+1)%2 == 0:
            ssim1_list = []
            ssim2_list = []
            ssim3_list = []
            print("----------- starting testing ----------")
            net.eval() 
            for i, batch_value in enumerate(test_loader):
                if i%2 == 0:

                    input1_tensor = batch_value[0].float()
                    input2_tensor = batch_value[1].float()

                    if torch.cuda.is_available():
                        input1_tensor = input1_tensor.cuda()
                        input2_tensor = input2_tensor.cuda()

                    with torch.no_grad():
                        batch_out = build_model(net, input1_tensor, input2_tensor, is_training=False)

                    output_H = batch_out['output_H'] 
                    output_H_ref = batch_out['output_H_ref']
                    output_H_tgt = batch_out['output_H_tgt']
                    output_tps_ref = batch_out['output_tps_ref']
                    output_tps_tgt = batch_out['output_tps_tgt']

                    # SSIM 1 
                    input1_np = ((input1_tensor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0) 
                    output = ((output_H[0,0:3,...]+1)*127.5).cpu().detach().numpy().transpose(1,2,0) 
                    overlap_mask = output_H[0,3:6,...].cpu().detach().numpy().transpose(1,2,0) 
                    # ssim1 = skimage.measure.compare_ssim(input1_np*overlap_mask, output*overlap_mask, data_range=255, multichannel=True) 

                    print('input1_np shape = ',np.shape(input1_np))
                    print('output shape = ',np.shape(output))
                    print('overlap_mask shape = ',np.shape(overlap_mask))
                    
                    print('input1_np*overlap_mask shape = ',np.shape(input1_np*overlap_mask))
                    
    
                    ssim1 = compare_ssim(input1_np*overlap_mask, output*overlap_mask, data_range = 255, channel_axis=-1) 
                    # SSIM 2
                    output_ref = ((output_H_ref[0,0:3,...]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                    output_tgt = ((output_H_tgt[0,0:3,...]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                    overlap_mask = output_H_ref[0,3:6,...] * output_H_tgt[0,3:6,...]
                    overlap_mask = overlap_mask.cpu().detach().numpy().transpose(1,2,0)
                    # ssim2 = skimage.measure.compare_ssim(output_ref*overlap_mask, output_tgt*overlap_mask, data_range=255, multichannel=True)

                    print('output_ref shape = ',np.shape(output_ref))
                    print('output_tgt shape = ',np.shape(output_tgt))
                    
                    ssim2 = compare_ssim(output_ref*overlap_mask, output_tgt*overlap_mask, data_range = 255, channel_axis=-1)

                    # SSIM 3
                    output_ref = ((output_tps_ref[0,0:3,...]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                    output_tgt = ((output_tps_tgt[0,0:3,...]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                    overlap_mask = output_tps_ref[0,3:6,...] * output_tps_tgt[0,3:6,...]
                    overlap_mask = overlap_mask.cpu().detach().numpy().transpose(1,2,0)
                    # ssim3 = skimage.measure.compare_ssim(output_ref*overlap_mask, output_tgt*overlap_mask, data_range=255, multichannel=True)

                    ssim3 = compare_ssim(output_ref*overlap_mask, output_tgt*overlap_mask, data_range = 255, channel_axis=-1) 

                    ssim1_list.append(ssim1)
                    ssim2_list.append(ssim2)
                    ssim3_list.append(ssim3)

            writer.add_scalar('SSIM1', np.mean(ssim1_list), epoch+1)
            writer.add_scalar('SSIM2', np.mean(ssim2_list), epoch+1)
            writer.add_scalar('SSIM3', np.mean(ssim3_list), epoch+1)

    print("##################end training#######################")


if __name__=="__main__":

    print('<==================== setting arguments ===================>\n')

    print('skimage-',skimage.__version__)

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--train_path', type=str, default='/paddle/yun/data/train/UDIS-D/') # training dataset
    parser.add_argument('--test_path', type=str, default='/paddle/yun/data/test/UDIS-D/testing/') # testing dataset

    args = parser.parse_args()
    print(args)
    train(args)