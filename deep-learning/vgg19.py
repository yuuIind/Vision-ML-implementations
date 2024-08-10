""" Vgg19

A PyTorch implementation of Vgg19

Paper: Very Deep Convolutional Networks for Large-Scale Image Recognition
- https://arxiv.org/abs/1409.1556

Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for 
large-scale image recognition. arXiv preprint arXiv:1409.1556.

@misc{simonyan2015deepconvolutionalnetworkslargescale,
      title={Very Deep Convolutional Networks for Large-Scale Image Recognition}, 
      author={Karen Simonyan and Andrew Zisserman},
      year={2015},
      eprint={1409.1556},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1409.1556}, 
}
"""

import torch 
import torch.nn as nn
# import torch.nn.functional as F

class VGG19(nn.Module):
    def __init__(self, num_classes=1000, p=0.5):
        super(VGG19, self).__init__()
        # input: (N,3,224,224)
        self.feature_extractor = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 64, (3,3), 1, 1), # (N,64,224,224)
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3), 1, 1), # (N,64,224,224)
            nn.ReLU(),
            nn.MaxPool2d(2,2), # (N,64,112,112)
            # Stage 2
            nn.Conv2d(64, 128, (3,3), 1, 1), # (N,128,112,112)
            nn.ReLU(),
            nn.Conv2d(128, 128, (3,3), 1, 1), # (N,64,112,112)
            nn.ReLU(),
            nn.MaxPool2d(2,2), # (N,128,56,56)
            # Stage 3
            nn.Conv2d(128, 256, (3,3), 1, 1), # (N,256,56,56)
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), 1, 1), # (N,256,56,56)
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), 1, 1), # (N,256,56,56)
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), 1, 1), # (N,256,56,56)
            nn.ReLU(),
            nn.MaxPool2d(2,2), # (N,256,28,28)
            # Stage 4
            nn.Conv2d(256, 512, (3,3), 1, 1), # (N,512,28,28)
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), 1, 1), # (N,512,28,28)
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), 1, 1), # (N,512,28,28)
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), 1, 1), # (N,512,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2,2), # (N,512,14,14)
            # Stage 5
            nn.Conv2d(512, 512, (3,3), 1, 1), # (N,512,14,14)
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), 1, 1), # (N,512,14,14)
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), 1, 1), # (N,512,14,14)
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), 1, 1), # (N,512,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2,2), # (N,512,7,7)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096), # (N,4096)
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4096, 4096), # (N,4096)
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4096, num_classes), # (N,C)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1, -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = VGG19(1000)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Model output shape: {out.shape}')
    print('VGG16')
    print(model)

