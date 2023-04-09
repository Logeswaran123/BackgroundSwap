import torch
from torch.autograd import Variable

import operations.utils as utils

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

CONTENT_WEIGHT = 1e5
STYLE_WEIGHT = 3e4
TV_WEIGHT = 1e0

NUM_EPOCHS = 1000


def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss + TV_WEIGHT * tv_loss

    return total_loss, content_loss, style_loss, tv_loss


def tuning_step(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, optimizer):
    total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices)
    # Computes gradients
    total_loss.backward()
    # Updates parameters and zeroes gradients
    optimizer.step()
    optimizer.zero_grad()
    return total_loss, content_loss, style_loss, tv_loss


class NeuralStyleTransfer():
    def __init__(self,):
        pass

    def perform(self,
                content,
                style, # background image passed from user is taken as the style to be applied to content image.
                ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        content = utils.prepare_img(content, device)
        style = utils.prepare_img(style, device)

        init_img = content

        optimizing_img = Variable(init_img, requires_grad=True)

        neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(device)

        content_img_set_of_feature_maps = neural_net(content)
        style_img_set_of_feature_maps = neural_net(style)

        target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
        target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
        target_representations = [target_content_representation, target_style_representation]

        optimizer = torch.optim.Adam((optimizing_img,), lr=1e1)

        torch.cuda.empty_cache()
        for epoch in range(NUM_EPOCHS):
            total_loss, _, _, _ = tuning_step(neural_net,
                                            optimizing_img,
                                            target_representations,
                                            content_feature_maps_index_name[0],
                                            style_feature_maps_indices_names[0],
                                            optimizer)
            with torch.no_grad():
                print(f"iteration: {epoch}, total loss={total_loss.item():12.4f}")

        stylized_image = utils.postprocess_image(optimizing_img)
        torch.cuda.empty_cache()

        return stylized_image
