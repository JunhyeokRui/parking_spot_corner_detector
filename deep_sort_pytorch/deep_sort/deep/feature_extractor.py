import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
from torchvision.models import resnet18, resnet50

from .model import Net, CustomNet
import torch.nn as nn


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)



class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
#         self.net = Net(reid=True)
#         self.net = CustomNet()
#         self.net = ResNetSimCLR('resnet18', 512)
        self.net = resnet18()
        self.net.fc = torch.nn.Identity()
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
#         state_dict = torch.load(model_path, map_location=torch.device(self.device))['net_dict']
        state_dict = torch.load(model_path, map_location=torch.device(self.device))
    
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 64)   #  aspect ratio is not optimal for corner points
        self.norm = transforms.Compose([
            transforms.ToTensor(),
#             transforms.CenterCrop([14,14]),
#             transforms.Resize([64,64])
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255.0, size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
