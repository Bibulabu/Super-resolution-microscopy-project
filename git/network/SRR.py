#------------------------------------------ MYMODEL ------------------------------------------------------#
#------------------------------------------ MYMODEL ------------------------------------------------------#
#------------------------------------------ MYMODEL ------------------------------------------------------#
class Encoder(nn.Module):
  def __init__(self, out_channels, n_resgroups):
    super(Encoder, self).__init__()
    
    # Step1 resgroup 尺寸不变
    m_body = [
            ResidualGroup(
                conv, n_feat=out_channels, kernel_size=3, reduction=16, act=nn.ReLU(True), res_scale=1, n_resblocks=3) \
            for _ in range(n_resgroups)]
    
    m_body.append(conv(out_channels, out_channels, kernel_size=3))
    self.res = nn.Sequential(*m_body)

  def forward(self, x):
    res = self.res(x)
    res = res + x
    return res

class Down(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Down, self).__init__()
    # Step2 pooling 尺寸1/2
    self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))

  def forward(self, x):
    x = self.down(x)
    return x


class Decoder(nn.Module):
  def __init__(self, pre_cha, in_channels, mid_channels, out_channels):
    super(Decoder, self).__init__()
    # Step1 反卷积 尺寸x2
    self.up = nn.ConvTranspose2d(pre_cha, pre_cha, kernel_size=2, stride=2) #尺寸x2
    # Step2 尺寸不变
    self.conv3_relu = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )

    # Step3 channel attention 尺寸不变
    self.calayer = nn.Sequential(CALayer(out_channels, reduction=16))

  def forward(self, x1, x2):
    x1 = self.up(x1)
    x = torch.cat((x1, x2), dim=1)
    
    x = self.conv3_relu(x)

    x_ca = self.calayer(x)

    return x_ca




class MYMODEL(nn.Module):
    def __init__(self, opt):
        super(MYMODEL, self).__init__()
        self.head = nn.Sequential(
        nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True)
        )
        self.en1 = Encoder(out_channels=32, n_resgroups=5)
        self.o1 = Down(in_channels=32, out_channels=64)
        self.en2 = Encoder(out_channels=64, n_resgroups=5)
        self.o2 = Down(in_channels=64, out_channels=128)
        self.en3 = Encoder(out_channels=128, n_resgroups=5)
        self.o3 = Down(in_channels=128, out_channels=256)
        self.en4 = Encoder(out_channels=256, n_resgroups=5)


        self.de3 = Decoder(pre_cha=256, in_channels=256+128, mid_channels=256, out_channels=128)
        self.de2 = Decoder(pre_cha=128, in_channels=128+64, mid_channels=128, out_channels=64)
        self.de1 = Decoder(pre_cha=64, in_channels=64+32, mid_channels=64, out_channels=32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 9, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(9, opt.nch_out, kernel_size=3, padding=1),
            )

    def forward(self, input): 
        #input 512x512x9
        head = self.head(input)#512x512x32

        e1 = self.en1(head) #512x512x32
        e1_down = self.o1(e1) #256x256x64
        e2 = self.en2(e1_down) #256x256x64
        e2_down = self.o2(e2) #128x128x128
        e3 = self.en3(e2_down) #128x128x128
        e3_down = self.o3(e3) #64x64x256
        e4 = self.en4(e3_down) #64x64x256

        d3 = self.de3(e4, e3) #64x64x128
        d2 = self.de2(d3, e2) #128x128x64
        d1 = self.de1(d2, e1) #256x256x32

        final = self.final(d1) #512x512x9
        final = input + final #512x512x9
        output = self.out(final) #512x512x1
        return output
