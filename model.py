import timm
import torch
import torch.nn as nn

from CA_block import resnet18_pos_attention
from callbacks import gen_state_dict
from magnet import MagNet

def create_model(num_classes=7, is_of=False, is_motion=False):
    model = ResNet18(num_classes=num_classes, is_of=is_of, is_motion=is_motion)

    return model

class ResNet18(nn.Module):
    def __init__(self, num_classes=7, is_of=False, is_motion=False):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.is_of = is_of
        self.is_motion = is_motion

        if is_of and is_motion:
            conv_of = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=180, kernel_size=3, stride=2,padding=1, bias=False, groups=1),
                                 nn.BatchNorm2d(180),
                                 nn.ReLU(inplace=True))
            feature_of = resnet18_pos_attention()
            self.feature_of = nn.Sequential(conv_of, feature_of)

            self.motion = Motion()
            for param in self.motion.parameters():
                param.requires_grad = False
            conv_motion = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=180, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
                                 nn.BatchNorm2d(180),
                                 nn.ReLU(inplace=True))
            feature_motion = resnet18_pos_attention()
            self.feature_motion = nn.Sequential(conv_motion, feature_motion)

            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512 * 196, num_classes))
            
        elif is_of:
            conv = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=180, kernel_size=3, stride=2, padding=1, bias=False, groups=1),
                                 nn.BatchNorm2d(180),
                                 nn.ReLU(inplace=True))
            feature = resnet18_pos_attention()
            classifier = nn.Sequential(nn.Flatten(), nn.Linear(512 * 196, num_classes))
            self.model = nn.Sequential(conv, feature, classifier)
        elif is_motion:
            self.motion = Motion()
            for param in self.motion.parameters():
                param.requires_grad = False
            conv = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=180, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
                                 nn.BatchNorm2d(180),
                                 nn.ReLU(inplace=True))
            feature = resnet18_pos_attention()
            classifier = nn.Sequential(nn.Flatten(), nn.Linear(512 * 196, num_classes))
            self.model = nn.Sequential(conv, feature, classifier)
        else:
            conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=180, kernel_size=3, stride=2, padding=1, bias=False, groups=1),
                                 nn.BatchNorm2d(180),
                                 nn.ReLU(inplace=True))
            feature = resnet18_pos_attention()
            classifier = nn.Sequential(nn.Flatten(), nn.Linear(512 * 196, num_classes))
            self.model = nn.Sequential(conv, feature, classifier)

    def forward(self, images_on, images_apex, of_u, of_v):
        if self.is_of and self.is_motion:
            of = torch.cat((of_u, of_v), 1)
            out_of = self.feature_of(of)
            out_motion = self.motion(images_on, images_apex)
            out_motion = self.feature_motion(out_motion)
            out = out_of + out_motion
            out = self.classifier(out)
        elif self.is_of:
            out = torch.cat((of_u, of_v), 1)
            out = self.model(out)
        elif self.is_motion:
            out = self.motion(images_on, images_apex)
            out = self.model(out)
        else:
            out = self.model(images_apex)

        return out

class Motion(nn.Module):
    def __init__(self):
        super(Motion, self).__init__()
        magnet = MagNet()

        weights_path = 'magnet_epoch12_loss7.28e-02.pth'
        state_dict = gen_state_dict(weights_path)
        magnet.load_state_dict(state_dict)
        for param in magnet.parameters():
            param.requires_grad = False

        self.encoder = magnet.encoder

    def forward(self, batch_A, batch_B):
        texture_A, motion_A = self.encoder(batch_A)
        texture_B, motion_B = self.encoder(batch_B)

        return motion_B - motion_A