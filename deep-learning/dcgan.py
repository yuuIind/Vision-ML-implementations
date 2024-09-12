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
    def __init__(self, dim_z=100, dim_c=3, dim_hid=128):
        super(Generator, self).__init__()
        self.block1 = self._block(c_in=dim_z, c_out=dim_hid*8, k=4)
        self.block2 = self._block(c_in=dim_hid*8, c_out=dim_hid*4, k=4, s=2, p=1)
        self.block3 = self._block(c_in=dim_hid*4, c_out=dim_hid*2, k=4, s=2, p=1)
        self.block4 = self._block(c_in=dim_hid*2, c_out=dim_hid, k=4, s=2, p=1)
        self.block5 = self._block(c_in=dim_hid, c_out=dim_c, k=4, s=2, p=1, is_final=True)

    def _block(self, c_in, c_out, k, s=1, p=0, is_final=False):
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
        x = self.block1(x) # (N,100,1,1) -> (N,1024,4,4)
        x = self.block2(x) # (N,1024,4,4) -> (N,512,8,8)
        x = self.block3(x) # (N,512,8,8) -> (N,256,16,16)
        x = self.block4(x) # (N,256,16,16) -> (N,128,32,32)
        x = self.block5(x) # (N,128,32,32) -> (N,3,64,64)
        return x


class Discriminator(nn.Module):
    def __init__(self, c_in=3, c_out=1, dim_hid=64, slope=0.2):
        super(Discriminator, self).__init__()
        self.block1 = nn.Sequential( # The first layer of Discriminator did not use BN
                nn.Conv2d(
                    in_channels=c_in, out_channels=dim_hid,
                    kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=slope)
        )
        self.block2 = self._block(
            c_in=dim_hid, c_out=dim_hid*2, 
            k=4, s=2, p=1, slope=slope
            )
        self.block3 = self._block(
            c_in=dim_hid*2, c_out=dim_hid*4, 
            k=4, s=2, p=1, slope=slope
            )
        self.block4 = self._block(
            c_in=dim_hid*4, c_out=dim_hid*8, 
            k=4, s=2, p=1, slope=slope
            )
        self.block5 = self._block(
            c_in=dim_hid*8, c_out=c_out, 
            k=4, s=2, is_final=True
            )

    def _block(self, c_in, c_out, k, s=1, p=0, slope=0.2, is_final=False):
        if is_final:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=c_in, out_channels=c_out,
                    kernel_size=k, stride=s, padding=p
                    ),
                nn.Sigmoid(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=c_in, out_channels=c_out,
                    kernel_size=k, stride=s, padding=p
                    ),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(negative_slope=slope),
            )

    def forward(self, x):
        x = self.block1(x) # (N,3,64,64) -> (N,64,32,32)
        x = self.block2(x) # (N,64,32,32) -> (N,128,16,16)
        x = self.block3(x) # (N,128,16,16) -> (N,256,8,8)
        x = self.block4(x) # (N,256,8,8) -> (N,512,4,4)
        x = self.block5(x) # (N,512,4,4) -> (N,1,1,1)
        return x
    
class DCGAN():
    def __init__(self, dim_z=100, dim_c=3, dim_g=128, dim_d=64, slope=0.2, init_def=True):
        self.gen = Generator(dim_z=dim_z, dim_c=dim_c, dim_hid=dim_g)
        self.disc = Discriminator(c_in=dim_c, c_out=1, dim_hid=dim_d, slope=slope)

        if init_def:
            self.gen = self.gen.apply(DCGAN.weights_init)
            self.disc = self.disc.apply(DCGAN.weights_init)


    def weights_init(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    
    def _test(self):
        z = torch.randn(5, 100, 1, 1)

        fake = self.gen(z)
        print('Generator works well!')
        
        pred = self.disc(fake)
        print('Discriminator works well!')

        print('DCGAN works well!')
        print(f'Latent vector z shape: {z.shape}')
        print(f'Generated image shape: {fake.shape}')
        print(f'Discriminator prediction shape: {pred.shape}')

if __name__ == "__main__":
    # Test the model with random input
    dcgan = DCGAN()

    # Print model output
    dcgan._test()
    
    # Print the number of parameters in the model
    total_params = sum(p.numel() for p in dcgan.gen.parameters())
    total_trainable_params = sum(p.numel() for p in dcgan.gen.parameters() if p.requires_grad)
    print('Generator:')
    print(f"Total params: {total_params}, Total trainable params: {total_trainable_params}\n")
    
    # Print the number of parameters in the model
    total_params = sum(p.numel() for p in dcgan.disc.parameters())
    total_trainable_params = sum(p.numel() for p in dcgan.disc.parameters() if p.requires_grad)
    print('Discriminator')
    print(f"Total params: {total_params}, Total trainable params: {total_trainable_params}\n")

    print_model_summary = False
    if print_model_summary:
        # Print model summary
        print('DCGAN Generator')
        print(dcgan.gen)

        # Print model summary
        print('DCGAN Discriminator')
        print(dcgan.disc)