""" DCGAN

A PyTorch implementation of DCGAN

Paper: Unsupervised Representation Learning with Deep Convolutional 
Generative Adversarial Networks
- https://arxiv.org/abs/1511.06434

Radford, A. (2015). Unsupervised representation learning with deep 
convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.

@misc{radford2016unsupervisedrepresentationlearningdeep,
      title={Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks}, 
      author={Alec Radford and Luke Metz and Soumith Chintala},
      year={2016},
      eprint={1511.06434},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1511.06434}, 
}
"""

import torch 
import torch.nn as nn
# import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, dim_z=100, dim_hid=128, dim_c=3, default_init=False):
        super(Generator, self).__init__()
        self.block1 = self.gen_block(c_in=dim_z, c_out=dim_hid*8, k=4)
        self.block2 = self.gen_block(c_in=dim_hid*8, c_out=dim_hid*4, k=4, s=2, p=1)
        self.block3 = self.gen_block(c_in=dim_hid*4, c_out=dim_hid*2, k=4, s=2, p=1)
        self.block4 = self.gen_block(c_in=dim_hid*2, c_out=dim_hid, k=4, s=2, p=1)
        self.block5 = self.gen_block(c_in=dim_hid, c_out=dim_c, k=4, s=2, p=1, is_final=True)

    def gen_block(self, c_in, c_out, k, s=1, p=0, is_final=False):
        if is_final:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=c_in, out_channels=c_out,
                    kernel_size=k, stride=s, padding=p
                    ),
                nn.Tanh(),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=c_in, out_channels=c_out,
                    kernel_size=k, stride=s, padding=p
                    ),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.block1(x) # (N,100,1,1) -> # (N,1024,4,4)
        x = self.block2(x) # (N,1024,4,4) -> # (N,512,8,8)
        x = self.block3(x) # (N,512,8,8) -> # (N,256,16,16)
        x = self.block4(x) # (N,256,16,16) -> # (N,128,32,32)
        x = self.block5(x) # (N,128,32,32) -> # (N,3,64,64)
        return x


class Discriminator(nn.Module):
    def __init__(self, dim_z=100, dim_c=3, default_init=False):
        super(Discriminator, self).__init__()

    def forward(x):
        pass


if __name__ == "__main__":
    # Test the model with random input
    gen = Generator()
    x = torch.randn(5, 100, 1, 1)
    out = gen(x)

    # Print model output
    print(f'Input shape: {x.shape}')
    print(f'Model output shape: {out.shape}\n')

    """    
    # Print the number of parameters in the model
    total_params = sum(p.numel() for p in gen.parameters())
    total_trainable_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"Total params: {total_params}, Total trainable params: {total_trainable_params}\n")

    # Print model summary
    print('DCGAN Generator')
    print(gen)

    # Test the model with random input
    disc = Discriminator()
    x = torch.randn(1, 3, 224, 224)
    out = disc(x)

    # Print model output
    print(f'Input shape: {x.shape}')
    print(f'Model output shape: {out.shape}\n')
    
    # Print the number of parameters in the model
    total_params = sum(p.numel() for p in disc.parameters())
    total_trainable_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Total params: {total_params}, Total trainable params: {total_trainable_params}\n")

    # Print model summary
    print('DCGAN Discriminator')
    print(disc)"""