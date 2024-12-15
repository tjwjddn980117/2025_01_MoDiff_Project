import torch
import torch.nn.functional as F
import os
import logging
import random
import numpy as np
import torch.backends.cudnn as cudnn

def random_init(seed=0):
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger

def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0

    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

class m_scheduler:
    def __init__(self, module, param_name, param_list, patience, logger):
        self.param_module = module
        self.param_name = param_name
        self.param_list = param_list
        self.patience = patience
        self.best = np.inf
        self.stop_updating = False
        self.current_index = 0
        self.logger = logger

    def reset(self, metric):
        self.best = metric
        self.num_bad_epochs = 0

    def update_param(self):
        if self.current_index < len(self.param_list):
            setattr(self.param_module, self.param_name, self.param_list[self.current_index])
            # self.param_module.gamma = self.param_list[self.current_index]
            self.current_index += 1
            return True
        else:
            return False

    def step(self, metric):
        if self.stop_updating:
            return

        if metric < self.best:
            self.reset(metric)
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            if self.update_param():
                self.logger.warning("Update param {} to {}! ".format(self.param_name,
                                                       getattr(self.param_module, self.param_name)))
                self.reset(np.inf)
            else:
                self.stop_updating = True

        return

def get_array_of_modes(seg):
    """
    Assemble an array holding all label modes.
    :param cf: config module
    :param seg: 4D integer array
    :return: 4D integer array
    """
    exp_modes = len(sys_config.label_switches.keys())
    mode_stats = get_mode_statistics(exp_modes)
    switch = mode_stats['switch']
    num_modes = 2**exp_modes

    # construct ground-truth modes
    gt_seg_modes = np.zeros(shape=(num_modes,) + seg.shape, dtype=np.uint8)
    for mode in range(num_modes):
        switched_seg = seg.copy()
        for i, c in enumerate(sys_config.label_switches.keys()):
            if switch[mode, i]:
                init_id = sys_config.name2trainId[c]
                final_id = sys_config.name2trainId[c + '_2']

                switched_seg[switched_seg == init_id] = final_id
        gt_seg_modes[mode] = switched_seg

    return gt_seg_modes, mode_stats['mode_probs']

def get_conf_map_accumulate(prediction, prob, masks_arrangement, prob_gt, n_classes,all_pixel_conf, all_gt_conf, all_pixel_image):

    # Prediction can either be logit or softmax.
    if ((prediction > 1.)).any():
        prediction = F.softmax(prediction, dim=2)

    pixel_image = torch.sum(prediction * prob.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), axis=1).detach().cpu()

    # Calculate the marginalized pixel-wise probabilities for both model outputs and ground truth labels.
    pixel_conf = torch.sum(prediction * prob.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                                     axis=1).transpose(0, 1).flatten(1).detach().cpu()  # B, CL, H, W -> CL, B,HxW
    masks_onehot = F.one_hot(masks_arrangement.long(), n_classes).permute(0, 1, -1, 2, 3)  # B,M,CL,H,W
    gt_conf = torch.sum(masks_onehot.float() * prob_gt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                          axis=1).transpose(0, 1).flatten(1).detach().cpu()

    # Get accumulated results.
    if all_pixel_conf is None:
        all_pixel_conf, all_gt_conf = pixel_conf, gt_conf
        all_pixel_image = pixel_image
    else:
        all_pixel_conf = torch.cat([all_pixel_conf,pixel_conf], 1)
        all_gt_conf = torch.cat([all_gt_conf,gt_conf], 1)
        all_pixel_image = torch.cat([all_pixel_image,pixel_image], dim=0)

    return all_pixel_conf, all_gt_conf, all_pixel_image

def get_conf_map_accumulate_with_16_sampling(prediction, all_pixel_image):

    # Prediction can either be logit or softmax.
    if ((prediction > 1.)).any():
        prediction = F.softmax(prediction, dim=2)

    pixel_image = prediction.cpu().detach()

    # Get accumulated results.
    if all_pixel_image is None:
        all_pixel_image = pixel_image
    else:
        all_pixel_image = torch.cat([all_pixel_image,pixel_image], dim=0)

    return all_pixel_image

def get_16_sampling(prediction):
    # Prediction can either be logit or softmax.
    if ((prediction > 1.)).any():
        prediction = F.softmax(prediction, dim=2)

    pixel_image = prediction.cpu().detach()

    return pixel_image


def get_model_size(model):
    size_model = 0
    for param in model.parameters():
        if param.requires_grad:
            size_model += param.numel()
    size_model = size_model * 4 * 1e-6

    return size_model
import torch
import os
from PIL import Image

def save_pixel_images(pixel_image, output_dir, threshold=None):
    """
    pixel_image: torch.Tensor with shape [B, C, H, W]
    output_dir: directory where images will be saved
    class_idx: index of the class to save as an image (optional)
    threshold: if set, values above threshold will be kept, others will be set to 0 (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+'/numpy', exist_ok=True)
    os.makedirs(output_dir+'/fill_unmask', exist_ok=True)
    os.makedirs(output_dir+'/fill_mask', exist_ok=True)
    os.makedirs(output_dir+f'/fill_unmask_threshold_{threshold}', exist_ok=True)
    os.makedirs(output_dir+f'/fill_mask_threshold_{threshold}', exist_ok=True)
    os.makedirs(output_dir+'/argmax', exist_ok=True)

    # 1. Save the whole pixel_image as a numpy file
    np_pixel_image = pixel_image.cpu().numpy()
    np.save(os.path.join(output_dir,'numpy', 'pixel_image.npy'), np_pixel_image)
    #print(f"Saved full pixel_image as numpy file: {os.path.join(output_dir,'numpy', 'pixel_image.npy')}")

    # Iterate through each image in the batch
    # for i in range(pixel_image.shape[0]):  # B (batch dimension)
    #     if class_idx is not None:
    #         # 2. Save only a specific class index as image
    #         class_image = pixel_image[i, class_idx].cpu().numpy()
    #         class_image_path = os.path.join(output_dir, f'class_{class_idx}_image_{i}.png')
    #         Image.fromarray((class_image * 255).astype(np.uint8)).save(class_image_path)
    #         print(f'Saved class {class_idx} image for batch {i}: {class_image_path}')

    #         # 3. Apply threshold if specified and save
    #         if threshold is not None:
    #             thresholded_image = np.where(class_image > threshold, class_image, 0)
    #             threshold_image_path = os.path.join(output_dir, f'class_{class_idx}_threshold_{threshold}_image_{i}.png')
    #             Image.fromarray((thresholded_image * 255).astype(np.uint8)).save(threshold_image_path)
    #             print(f'Saved thresholded class {class_idx} image for batch {i}: {threshold_image_path}')
    #     else:
    #         # Save all classes for each batch if no specific class_idx is provided
    #         for c in range(pixel_image.shape[1]):  # C (class dimension)
    #             class_image = pixel_image[i, c].cpu().numpy()
    #             class_image_path = os.path.join(output_dir, f'class_{c}_image_{i}.png')
    #             Image.fromarray((class_image * 255).astype(np.uint8)).save(class_image_path)
    #             print(f'Saved class {c} image for batch {i}: {class_image_path}')

    #     # 4. Save argmax image (index of the class with highest probability)
    #     argmax_image = torch.argmax(pixel_image[i], dim=0).cpu().numpy()
    #     argmax_image_path = os.path.join(output_dir, f'argmax_image_{i}.png')
    #     Image.fromarray((argmax_image * 255 / (pixel_image.shape[1] - 1)).astype(np.uint8)).save(argmax_image_path)
    #     print(f'Saved argmax image for batch {i}: {argmax_image_path}')

    # Iterate through each image in the batch
    for i in range(pixel_image.shape[0]):  # B (batch dimension)
        for c in range(pixel_image.shape[1]):  # C (class dimension)
            if c == 0:
                class_image = pixel_image[i, c].cpu().numpy()
                class_image_path = os.path.join(output_dir,f'fill_unmask', f'class_fill_unmask_image_{i}.png')
                Image.fromarray((class_image * 255).astype(np.uint8)).save(class_image_path)
                #print(f"Saved class {c} image for batch {i}: {class_image_path}")

                if threshold is not None:
                    thresholded_image = np.where(class_image > threshold, class_image, 0)
                    threshold_image_path = os.path.join(output_dir,f'fill_unmask_threshold_{threshold}', f'class_fill_unmask_threshold_{threshold}_image{i}.png')
                    Image.fromarray((thresholded_image * 255).astype(np.uint8)).save(threshold_image_path)
            else:
                class_image = pixel_image[i, c].cpu().numpy()
                class_image_path = os.path.join(output_dir,f'fill_mask', f'class_fill_mask_image_{i}.png')
                Image.fromarray((class_image * 255).astype(np.uint8)).save(class_image_path)
                #print(f"Saved class {c} image for batch {i}: {class_image_path}")

                if threshold is not None:
                    thresholded_image = np.where(class_image > threshold, class_image, 0)
                    threshold_image_path = os.path.join(output_dir,f'fill_mask_threshold_{threshold}', f'class_fill_mask_threshold_{threshold}_image{i}.png')
                    Image.fromarray((thresholded_image * 255).astype(np.uint8)).save(threshold_image_path)


        # 4. Save argmax image (index of the class with highest probability)
        argmax_image = torch.argmax(pixel_image[i], dim=0).cpu().numpy()
        argmax_image_path = os.path.join(output_dir,'argmax', f'argmax_image_{i}.png')
        Image.fromarray((argmax_image * 255 / (pixel_image.shape[1] - 1)).astype(np.uint8)).save(argmax_image_path)
        #print(f"Saved argmax image for batch {i}: {argmax_image_path}")

def save_pixel_images_for_16samples(pixel_image, output_dir, threshold=None):
    """
    pixel_image: torch.Tensor with shape [B, C, H, W]
    output_dir: directory where images will be saved
    class_idx: index of the class to save as an image (optional)
    threshold: if set, values above threshold will be kept, others will be set to 0 (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+'/numpy', exist_ok=True)
    os.makedirs(output_dir+'/fill_unmask', exist_ok=True)
    os.makedirs(output_dir+'/fill_mask', exist_ok=True)
    os.makedirs(output_dir+f'/fill_unmask_threshold_{threshold}', exist_ok=True)
    os.makedirs(output_dir+f'/fill_mask_threshold_{threshold}', exist_ok=True)
    os.makedirs(output_dir+'/argmax', exist_ok=True)

    # 1. Save the whole pixel_image as a numpy file
    np_pixel_image = pixel_image.cpu().numpy()
    np.save(os.path.join(output_dir,'numpy', 'pixel_image.npy'), np_pixel_image)

    # Iterate through each image in the batch
    for i in range(pixel_image.shape[0]):  # B (batch dimension)
        for c in range(pixel_image.shape[1]):  # C (class dimension)
            if c == 0:
                class_image = pixel_image[i, c].cpu().numpy()
                class_image_path = os.path.join(output_dir,f'fill_unmask', f'class_fill_unmask_image_{i}.png')
                Image.fromarray((class_image * 255).astype(np.uint8)).save(class_image_path)
                #print(f"Saved class {c} image for batch {i}: {class_image_path}")

                if threshold is not None:
                    thresholded_image = np.where(class_image > threshold, class_image, 0)
                    threshold_image_path = os.path.join(output_dir,f'fill_unmask_threshold_{threshold}', f'class_fill_unmask_threshold_{threshold}_image{i}.png')
                    Image.fromarray((thresholded_image * 255).astype(np.uint8)).save(threshold_image_path)
            else:
                class_image = pixel_image[i, c].cpu().numpy()
                class_image_path = os.path.join(output_dir,f'fill_mask', f'class_fill_mask_image_{i}.png')
                Image.fromarray((class_image * 255).astype(np.uint8)).save(class_image_path)
                #print(f"Saved class {c} image for batch {i}: {class_image_path}")

                if threshold is not None:
                    thresholded_image = np.where(class_image > threshold, class_image, 0)
                    threshold_image_path = os.path.join(output_dir,f'fill_mask_threshold_{threshold}', f'class_fill_mask_threshold_{threshold}_image{i}.png')
                    Image.fromarray((thresholded_image * 255).astype(np.uint8)).save(threshold_image_path)


        # 4. Save argmax image (index of the class with highest probability)
        argmax_image = torch.argmax(pixel_image[i], dim=0).cpu().numpy()
        argmax_image_path = os.path.join(output_dir,'argmax', f'argmax_image_{i}.png')
        Image.fromarray((argmax_image * 255 / (pixel_image.shape[1] - 1)).astype(np.uint8)).save(argmax_image_path)
        #print(f"Saved argmax image for batch {i}: {argmax_image_path}")

