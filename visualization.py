import os
import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch import nn, optim
from torchsummary import summary
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from LoadDataset import LoadDataset
from VIT_Modules import ViTPose, ClassicDecoder
from Loss import AdaptiveWingLoss, pose_pck_accuracy, _get_max_preds
from Utility import tensor_to_image


def transform_preds(coords,center,scale,output_size):
    scale = scale * 200.0

    scale_x = scale[0] / (output_size[0] - 1.0)
    scale_y = scale[1] / (output_size[1] - 1.0)
    
    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    
    return target_coords



if __name__ == "__main__":
    in_channels = 3
    patch_size = 16
    emb_size = 768
    img_size = (128,128)
    heatmap_size = ((img_size[0]//patch_size)*4, (img_size[1]//patch_size)*4)
    depth = 12                    #Depth of transformer layer
    kernel_size = (4,4)
    deconv_filter = 256
    out_channels = 17
    train_dataset_size = 90000
    val_dataset_size = 10000
    batch_size = 25

    learning_rate = 2e-3
    weight_decay = 1e-4
    train_max_iters = train_dataset_size//batch_size
    val_max_iters = val_dataset_size//batch_size

    log_period = 20
    num_epochs = 200
    device = "cuda"

    img_directory = "/home/biped-lab/504_project/coco/images/train2017/"
    annotation_path = "/home/biped-lab/504_project/coco/annotations/person_keypoints_train2017.json"
    model_save_path = "/home/biped-lab/504_project/coco/model/"
    print(torch.__version__)
    print(torch.cuda.is_available())
    
    train = LoadDataset(img_directory, annotation_path,img_size, heatmap_size, test_mode = False)
    indices = torch.arange(10000)
    train_30k= data_utils.Subset(train, indices)

    # print("TAKING SUBSET OF THE CURRENT DATASET")
    print("New Dataset Size: ",train_30k.__len__())
    train_loader = torch.utils.data.DataLoader(dataset=train_30k, 
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4)

    model = ViTPose(in_channels,patch_size,emb_size,img_size,depth,kernel_size,deconv_filter,out_channels)

    # train_ViTPose( model, train_loader, learning_rate, weight_decay, train_max_iters, log_period, num_epochs, device, model_save_path,use_checkpoint = True, val = False)
    
    trained_model = torch.load(os.path.join(model_save_path,"model_params1.pth"))
    model.load_state_dict(trained_model)
    model.to(device=device)
    
    iterator = iter(train_loader)
    images, target_heatmap, t_h_weight,center, scale, paths  = next(iterator)
    # print("paths shape: ", paths[0])
    # print("Scale Shape: ", scale.shape)
    t_h_weight = rearrange(t_h_weight, "B C H W ->  B H C W")
                    
    images = images.to(device)
    target_heatmap = target_heatmap.to(device)
    t_h_weight =  t_h_weight.to(device)

    model.train()
    model.zero_grad()

    generated_heatmaps = model(images)

    pred_keypoints, confidence = _get_max_preds(generated_heatmaps.detach().cpu().numpy())
    gt, _ = _get_max_preds(target_heatmap.detach().cpu().numpy())
    conf = confidence.copy()

    conf = conf.squeeze()
    conf = np.mean(conf,axis = 1)
    max_confidence_idx = np.argmax(conf)
    
    img = images[max_confidence_idx]
    kp = pred_keypoints[max_confidence_idx]  
    gtkp = gt[max_confidence_idx]
    c = center[max_confidence_idx]
    s = scale[max_confidence_idx]
    path = paths[max_confidence_idx]

    
    print(kp)
    print(c.squeeze())

    img = Image.open(os.path.normpath(path)).convert('RGB')
    img = np.array(img)
    cvImage = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    transformed_kps = transform_preds(kp,c.squeeze().detach().numpy(),s.squeeze().detach().numpy(),heatmap_size)
    print(transformed_kps)
    for i in range(17):
        (x,y) = transformed_kps[i]
        cvImage = cv2.circle(cvImage, (int(x),int(y)), 2, (255, 0, 0), 2)
    cv2.imshow("result",cvImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # res = Image.fromarray(res).convert("RGB")
    # res.show()
    
    # val_indices = torch.arange(90000, 100000)
    # val_10k= data_utils.Subset(train, val_indices)

    # val_loader = torch.utils.data.DataLoader(dataset=val_10k, 
    #                                             batch_size=batch_size,
    #                                             shuffle=False,
    #                                             num_workers=4)
    # print("Val Dataset Size: ",val_10k.__len__())

    # train_ViTPose( model, val_loader, learning_rate, weight_decay, val_max_iters, log_period, num_epochs, device, model_save_path,use_checkpoint = False,val = True)
    #print(summary(model,input_size=(in_channels, img_size[0], img_size[1])))
    # tensor, target, weight =  next(iter(val_loader))
    # # # weight = rearrange(weight, "B C H W ->  B H C W")
    # # print("Image size: ", tensor[0].shape)
    # # # print("heatmap size: ", target.shape)
    # # # print("Target weights: ",weight.shape)
    # image = tensor_to_image(tensor[5])
    # data = Image.fromarray(image)
    # data.show()
    # print(weight[5])