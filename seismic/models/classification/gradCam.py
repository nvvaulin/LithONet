import numpy as np

from gradcam import GradCAM
from gradcam.utils import visualize_cam


def show_gradcam(tensor, model):
    # Feedforward image, calculate GradCAM and gather heatmap
    if model.__class__.__name__ == 'ResNet':
        target_layer = model.layer4[-1].conv2
        gradcam = GradCAM(model, target_layer)

    elif model.__class__.__name__ == 'DenseNet':
        target_layer = model.features.norm5
        gradcam = GradCAM(model, target_layer)

    elif model.__class__.__name__ == 'DataParallel':
        target_layer = model.module.densenet121.features.norm5
        gradcam = GradCAM(model.module.densenet121, target_layer)

    else:
        raise ValueError('improper model')

    mask, _ = gradcam(tensor)
    heatmap, _ = visualize_cam(mask, tensor)

    # heatmap from torch.tensor to numpy.array
    mask = mask[0].permute(1, 2, 0).detach().cpu().numpy()
    heatmap = heatmap.permute(1, 2, 0).numpy()

    return heatmap, mask
