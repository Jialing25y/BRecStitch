import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W


def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))

# alignment loss ###########################################
def cal_lp_loss(input1, output_H, output_H_ref, output_H_tgt, output_tps_ref, output_tps_tgt):
    batch_size, _, img_h, img_w = input1.size()
    # # part one:
    # overlap = output_H[:,3:6,:,:]
    # lp_loss_1_1 = l_num_loss(input1*overlap, output_H[:,0:3,:,:]*overlap, 1)

    overlap = output_H_ref[:,3:6,:,:] * output_H_tgt[:,3:6,:,:]
    lp_loss_1_2 = l_num_loss(output_H_ref[:,0:3,:,:]*overlap, output_H_tgt[:,0:3,:,:]*overlap, 1)

    # lp_loss_1 = (lp_loss_1_1 + lp_loss_1_2) / 2.
    lp_loss_1 = lp_loss_1_2

    # # part two:
    overlap = output_tps_ref[:,3:6,:,:] * output_tps_tgt[:,3:6,:,:]
    lp_loss_2 = l_num_loss(output_tps_ref[:,0:3,:,:]*overlap, output_tps_tgt[:,0:3,:,:]*overlap, 1)


    lp_loss = 3. * lp_loss_1 + 1. * lp_loss_2

    return lp_loss

# Not used
def cal_lp_loss2(input1, warp_mesh, warp_mesh_mask):
    batch_size, _, img_h, img_w = input1.size()

    delta3 = ( torch.sum(warp_mesh, [2,3])  -   torch.sum(input1*warp_mesh_mask, [2,3]) ) /  torch.sum(warp_mesh_mask, [2,3])
    input1_newbalance = input1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_2 = l_num_loss(input1_newbalance*warp_mesh_mask, warp_mesh, 1)
    lp_loss =  1. * lp_loss_2

    return lp_loss

# Not used_useless
def overlap_brightness_loss(output_tps_ref, output_tps_tgt):
    overlap_mask = output_tps_ref[:,3:6,:,:] * output_tps_tgt[:,3:6,:,:]
    
    brightness_loss = l_num_loss(
        output_tps_ref[:,0:3,:,:] * overlap_mask, 
        output_tps_tgt[:,0:3,:,:] * overlap_mask, 
        1
    )
    
    return brightness_loss

# for fine-tuning ###########################################
def cal_lp_loss3(output_tps_ref, output_tps_tgt):
    overlap = output_tps_ref[:,3:6,:,:] * output_tps_tgt[:,3:6,:,:]

    # 计算重叠区域像素占总像素数的比例，作为权重
    overlap_ratio = torch.sum(overlap) / (overlap.shape[2] * overlap.shape[3]) 
    weight = torch.clamp(overlap_ratio, min=0.1, max=1.0)*10.0

    lp_loss = l_num_loss(output_tps_ref[:,0:3,:,:]*overlap, output_tps_tgt[:,0:3,:,:]*overlap, 1)
    
    return weight * lp_loss

# shape loss ###########################################
def inter_grid_loss(mesh):

    batch_size = mesh.shape[0]

    overlap = torch.ones(batch_size, grid_h, grid_w).cuda()

    # compute horizontal edges
    w_edges = mesh[:,:,0:grid_w,:] - mesh[:,:,1:grid_w+1,:] 
    cos_w = torch.sum(w_edges[:,:,0:grid_w-1,:] * w_edges[:,:,1:grid_w,:],3) / (torch.sqrt(torch.sum(w_edges[:,:,0:grid_w-1,:]*w_edges[:,:,0:grid_w-1,:],3))*torch.sqrt(torch.sum(w_edges[:,:,1:grid_w,:]*w_edges[:,:,1:grid_w,:],3)))
    delta_w_angle = 1 - cos_w
    delta_w_angle = delta_w_angle[:,0:grid_h,:] + delta_w_angle[:,1:grid_h+1,:]

    # compute vertical edges
    h_edges = mesh[:,0:grid_h,:,:] - mesh[:,1:grid_h+1,:,:]
    cos_h = torch.sum(h_edges[:,0:grid_h-1,:,:] * h_edges[:,1:grid_h,:,:],3) / (torch.sqrt(torch.sum(h_edges[:,0:grid_h-1,:,:]*h_edges[:,0:grid_h-1,:,:],3))*torch.sqrt(torch.sum(h_edges[:,1:grid_h,:,:]*h_edges[:,1:grid_h,:,:],3)))
    delta_h_angle = 1 - cos_h 
    delta_h_angle = delta_h_angle[:,:,0:grid_w] + delta_h_angle[:,:,1:grid_w+1]

    # on overlapping regions 
    depth_diff_w = (1-torch.abs(overlap[:,:,0:grid_w-1] - overlap[:,:,1:grid_w])) * overlap[:,:,0:grid_w-1]
    error_w = depth_diff_w * delta_w_angle

    depth_diff_h = (1-torch.abs(overlap[:,0:grid_h-1,:] - overlap[:,1:grid_h,:])) * overlap[:,0:grid_h-1,:]
    error_h = depth_diff_h * delta_h_angle

    return torch.mean(error_w) + torch.mean(error_h)

def intra_grid_loss(pts):

    max_w = 512/grid_w * 2
    max_h = 512/grid_h * 2

    delta_x = pts[:,:,1:grid_w+1,0] - pts[:,:,0:grid_w,0]
    delta_y = pts[:,1:grid_h+1,:,1] - pts[:,0:grid_h,:,1]

    loss_x = F.relu(delta_x - max_w)
    loss_y = F.relu(delta_y - max_h)
    loss = torch.mean(loss_x) + torch.mean(loss_y)


    return loss

# boundary loss
def boundary_rect_loss(mesh_ref, mesh_tgt):

    width_max = torch.maximum(torch.max(mesh_ref[...,0]), torch.max(mesh_tgt[...,0]))
    width_min = torch.minimum(torch.min(mesh_ref[...,0]), torch.min(mesh_tgt[...,0])) 
    height_max = torch.maximum(torch.max(mesh_ref[...,1]), torch.max(mesh_tgt[...,1]))
    height_min = torch.minimum(torch.min(mesh_ref[...,1]), torch.min(mesh_tgt[...,1]))

    mesh_trans_ref = torch.stack([mesh_ref[...,0]-width_min, mesh_ref[...,1]-height_min], 3)
    mesh_trans_tgt = torch.stack([mesh_tgt[...,0]-width_min, mesh_tgt[...,1]-height_min], 3)
    
    grid_h, grid_w = 11, 11
    
    boundary_indices = torch.tensor([
        *[(0, j) for j in range(grid_w + 1)],
        *[(i, grid_w) for i in range(1, grid_h + 1)],
        *[(grid_h, j) for j in range(grid_w - 1, -1, -1)],
        *[(i, 0) for i in range(grid_h - 1, 0, -1)]
    ], device=mesh_ref.device, dtype=torch.long)  # (44, 2)
    
    boundary1 = mesh_trans_ref[0, boundary_indices[:, 0], boundary_indices[:, 1]]  # (44, 2)
    boundary2 = mesh_trans_tgt[0, boundary_indices[:, 0], boundary_indices[:, 1]]  # (44, 2)
    
    def batch_point_in_polygon(points, polygon):

        N = points.shape[0]
        M = polygon.shape[0]
     
        points_expanded = points.unsqueeze(1).expand(N, M, 2)  # (N, M, 2)
        polygon_expanded = polygon.unsqueeze(0).expand(N, M, 2)  # (N, M, 2)

        x, y = points_expanded[:, :, 0], points_expanded[:, :, 1]  # (N, M)
        
        # The starting and ending points of an edge
        p1x, p1y = polygon_expanded[:, :, 0], polygon_expanded[:, :, 1]  # (N, M)
        p2x, p2y = polygon_expanded[:, :, 0], polygon_expanded[:, :, 1]  # (N, M)
        
        # cyclic shift
        p2x = torch.roll(p2x, shifts=-1, dims=1)
        p2y = torch.roll(p2y, shifts=-1, dims=1)

        y_min = torch.minimum(p1y, p2y)
        cond1 = y > y_min
        
        y_max = torch.maximum(p1y, p2y)
        cond2 = y <= y_max
        
        x_max = torch.maximum(p1x, p2x)
        cond3 = x <= x_max
        
        cond4 = p1y != p2y
        
        # Avoid division by zero
        denominator = p2y - p1y
        xinters = torch.where(
            denominator != 0,
            (y - p1y) * (p2x - p1x) / denominator + p1x,
            torch.full_like(x, float('inf'))
        )
        
        cond5 = (p1y == p2y) | (x <= xinters)
        
        ray_intersects = cond1 & cond2 & cond3 & cond4 & cond5
        
        intersection_counts = torch.sum(ray_intersects, dim=1)  # (N,)
        
        # An odd number of intersections indicate that the point is inside
        inside = (intersection_counts % 2) == 1
        
        return inside
    
    boundary1_outside_2 = ~batch_point_in_polygon(boundary1, boundary2)  # (44,)
    boundary2_outside_1 = ~batch_point_in_polygon(boundary2, boundary1)  # (44,)
    
    outer_boundary1 = boundary1[boundary1_outside_2]
    outer_boundary2 = boundary2[boundary2_outside_1]
    
    if outer_boundary1.shape[0] == 0 and outer_boundary2.shape[0] == 0:
        return torch.tensor(0.0, device=mesh_ref.device, requires_grad=True)

    outer_boundary = torch.cat([outer_boundary1, outer_boundary2], dim=0)  # (M, 2)

    # Calculate the minimum bounding rectangle
    x_coords = outer_boundary[:, 0]
    y_coords = outer_boundary[:, 1]
    
    rect_x = torch.min(x_coords)
    rect_y = torch.min(y_coords)
    rect_width = torch.max(x_coords) - rect_x
    rect_height = torch.max(y_coords) - rect_y
    
    dist_to_top = torch.abs(y_coords - rect_y)  
    dist_to_right = torch.abs(x_coords - (rect_x + rect_width))  
    dist_to_bottom = torch.abs(y_coords - (rect_y + rect_height))  
    dist_to_left = torch.abs(x_coords - rect_x)  
    
    distances = torch.stack([dist_to_top, dist_to_right, dist_to_bottom, dist_to_left], dim=1)  # (M, 4)
    min_distances = torch.min(distances, dim=1)[0]  # (M,)
    
    total_distance = torch.sum(min_distances)
    
    return total_distance 
