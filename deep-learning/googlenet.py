""" GoogleNet

A PyTorch implementation of GoogleNet

Paper: Going Deeper with Convolutions
- https://arxiv.org/abs/1409.1556

Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... 
& Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings 
of the IEEE conference on computer vision and pattern recognition (pp. 1-9).

@misc{szegedy2014goingdeeperconvolutions,
      title={Going Deeper with Convolutions}, 
      author={Christian Szegedy and Wei Liu and Yangqing Jia and Pierre 
        Sermanet and Scott Reed and Dragomir Anguelov and Dumitru Erhan and 
        Vincent Vanhoucke and Andrew Rabinovich
        },
      year={2014},
      eprint={1409.4842},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1409.4842}, 
}
"""

import torch 
import torch.nn as nn
# import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(cin, cout, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    

class InceptionBlock(nn.Module):
    def __init__(self, cin, c1x1, cin3x3, cout3x3, cin5x5, cout5x5, c_pooling):
        super(InceptionBlock, self).__init__()
        
        # 1x1 Conv
        self.conv1x1 = ConvBlock(cin, c1x1, kernel_size=1)
        
        # 1x1 Conv -> 3x3 Conv
        self.conv3x3 = nn.Sequential(
            ConvBlock(cin, cin3x3, kernel_size=1),
            ConvBlock(cin3x3, cout3x3, kernel_size=3, padding=1)
        )
        
        # 1x1 Conv -> 5x5 Conv
        self.conv5x5 = nn.Sequential(
            ConvBlock(cin, cin5x5, kernel_size=1),
            ConvBlock(cin5x5, cout5x5, kernel_size=5, padding=2)
        )

        # 3x3 Max pooling -> 1x1 Conv
        self.pooling = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(cin, c_pooling, kernel_size=1),
        )
    
    def forward(self, x):
        a = self.conv1x1(x)
        b = self.conv3x3(x)
        c = self.conv5x5(x)
        d = self.pooling(x)
        return torch.cat([a, b, c, d], dim=1)
    

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, p=0.7):
        super(AuxiliaryClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            ConvBlock(in_channels, 128, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, aux_clf=False):
        super(GoogleNet, self).__init__()

        self.use_aux_clf = aux_clf

        # Block 1
        # Input size (N,3,224,224)
        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3) # (N,64,112,112)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1) # (N,64,56,56)
        self.lrn1 = nn.LocalResponseNorm(5) # Don't know why but people use 5 for size

        # Block 2
        self.conv2 = ConvBlock(64, 64, kernel_size=1, stride=1, padding=0) # (N,64,56,56)
        self.conv3 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1) # (N,192,56,56)
        self.lrn2 = nn.LocalResponseNorm(5) # (N,192,56,56)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1) # (N,192,28,28)

        # Block 3
        self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32) # (N,256,28,28)
        self.inception_3b = InceptionBlock(256, 128, 128, 192, 92, 96, 64) # (N,480,28,28)
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1) # (N,480,14,14)

        # Block 4
        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64) # (N,512,14,14)
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64) # (N,512,14,14)
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64) # (N,512,14,14)
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64) # (N,528,14,14)
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128) # (N,832,14,14)
        self.pool4 = nn.MaxPool2d(3, stride=2, padding=1) # (N,832,7,7)

        # Block 5
        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128) # (N,832,7,7)
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128) # (N,1024,7,7)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1) # (N,1024,1,1)

        # Classifier
        self.dropout = nn.Dropout(0.4) # (N,1024,1,1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024, num_classes) # (N,num_classes)

        # Auxiliary Classifier
        if self.use_aux_clf:
            self.aux_clf1 = AuxiliaryClassifier(512, num_classes)
            self.aux_clf2 = AuxiliaryClassifier(528, num_classes)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.lrn1(x)

        # Block 2
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.lrn2(x)
        x = self.pool2(x)

        # Block 3
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.pool3(x)

        # Block 4
        x = self.inception_4a(x)
        if self.use_aux_clf:
            aux1 = self.aux_clf1(x) # Auxillary Classifier 1 branch here
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        if self.use_aux_clf:
            aux2 = self.aux_clf2(x) # Auxillary Classifier 2 branch here
        x = self.inception_4e(x)
        x = self.pool4(x)

        # Block 5
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)

        # Classifier
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)

        if self.use_aux_clf:
            return x, aux1, aux2
        return x
    
if __name__ == "__main__":
    model = GoogleNet(1000, aux_clf=True)
    x = torch.randn(1, 3, 224, 224)
    out, aux1, aux2 = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Model output shape: {out.shape}')
    print(f'Auxillary Classifier 1 output shape: {aux1.shape}')
    print(f'Auxillary Classifier 2 output shape: {aux2.shape}')
    print('GoogleNet with Auxillary Classifier')
    print(model)