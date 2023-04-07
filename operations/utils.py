import os
import numpy as np
import torch
from torchvision import transforms

from operations.models import Vgg19

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def create_dir(dir_path):
    """ Helper function to create a directory in given path """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def postprocess_image(optimizing_img):
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)  # swap channel
    out_img = np.copy(out_img)
    out_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    out_img = np.clip(out_img, 0, 255).astype('uint8')
    return out_img


def prepare_img(image, device):
    image = image.astype(np.float32)
    image /= 255.0

    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.mul(255)),
                    transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    img = transform(image).to(device).unsqueeze(0)

    return img


def prepare_model(device):
    model = Vgg19(requires_grad=False, show_progress=True)

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names

    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)

    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))