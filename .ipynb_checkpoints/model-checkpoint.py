import torch
import torch.nn as nn
import pdb

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def conv_layer(in_c, out_c, with_maxpool = True):
    if with_maxpool:
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3,3,3), padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d((2,2,2))
        )
    return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3,3,3), padding=1),
            nn.LeakyReLU()
        )

def conv_layer_seg(in_c, out_c, with_maxpool = True):
    if with_maxpool:
        return nn.Sequential(
            nn.Conv3d(in_c, out_c//2, kernel_size=(3,3,3), padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(out_c//2, out_c, kernel_size=(3,3,3), padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d((2,2,2))
        )
    return nn.Sequential(
            nn.Conv3d(in_c, out_c//2, kernel_size=(3,3,3), padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(out_c//2, out_c, kernel_size=(3,3,3), padding=1),
            nn.LeakyReLU()
        )

class CNN3d(nn.Module):
    def __init__(self):
        super(CNN3d, self).__init__()
        # conv layers
        # 128 x 128 x 64
        self.layer_1 = conv_layer(1, 64)     # 64 x 64 x 32
        self.layer_2 = conv_layer(64, 128)  # 32 x 32 x 16
        self.layer_3 = conv_layer(128, 256)  # 16 x 16 x 8
        self.dropout = nn.Dropout(p=0.15)
        # norm layers
        self.norm64 = nn.BatchNorm3d(64, track_running_stats=False)
        self.norm128 = nn.BatchNorm3d(128, track_running_stats=False)
        self.norm256 = nn.BatchNorm3d(256, track_running_stats=False)
        # fully connected layers
        self.linear1 = nn.Linear(256, 1)
        self.linear2 = nn.Linear(16*16*8, 1)
        # relu activation
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.norm64(self.layer_1(x))
        x = self.norm128(self.layer_2(x))
        x = self.norm256(self.layer_3(x))
        num_samples = x.shape[0]
        # pdb.set_trace()
        x = x.view(num_samples, 256, -1).transpose(1, 2)
        x = self.relu(self.linear1(x).squeeze(-1))
        # pdb.set_trace()
        x = self.linear2(x)
        return x

class Unet3d(nn.Module):
    def __init__(self):
        super(Unet3d, self).__init__()
        # conv layers
        # 128 x 128 x 64
        self.layer_1 = conv_layer_seg(1, 64)     # 64 x 64 x 32
        self.layer_2 = conv_layer_seg(64, 128)  # 32 x 32 x 16
        self.layer_3 = conv_layer_seg(128, 256)  # 16 x 16 x 8
        self.dropout = nn.Dropout(p=0.15)


        self.up_layer3 = conv_layer_seg(256, 128, with_maxpool = False)     # 64 x 64 x 32
        self.up_layer2 = conv_layer_seg(128, 64, with_maxpool = False)  # 32 x 32 x 16

        self.final_conv = conv_layer(64, 1, with_maxpool = False)

        # norm layers  
        self.norm64 = nn.BatchNorm3d(64, track_running_stats=False)
        self.norm128 = nn.BatchNorm3d(128, track_running_stats=False)
        self.norm256 = nn.BatchNorm3d(256, track_running_stats=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        out1 = self.norm64(self.layer_1(x))
        out2 = self.norm128(self.layer_2(out1))
        out3 = self.norm256(self.layer_3(out2))

        up_out3 = self.up_layer3(self.upsample(out3))
        up_out2 = self.up_layer2(self.upsample(up_out3 + out2))
        up_out1 = self.upsample(up_out2 + out1)

        final_out = self.final_conv(up_out1)
        return self.sigmoid(final_out)


class NonLocalSE(nn.Module):
    def __init__(self):
        super(NonLocalSE, self).__init__()
        # conv layers
        # 128 x 128 x 64
        self.layer_1 = conv_layer(1, 64, with_maxpool = True)     # 64 x 64 x 32   x 64
        self.layer_2 = conv_layer(64, 128, with_maxpool = True)  # 32 x 32 x 16    x 128
        self.layer_3 = conv_layer(128, 256, with_maxpool = True)  # 16 x 16 x 8    x 256
        self.dropout = nn.Dropout(p=0.15)

        # resize into 32.
        self.uniform_0 = conv_layer(1, 32, with_maxpool = False)     # 128 x 128 x 1 x 32
        self.uniform_1 = conv_layer(64, 32, with_maxpool = False)     # 64 x 64 x 32 x 32
        self.uniform_2 = conv_layer(128, 32, with_maxpool = False)  # 32 x 32 x 16 x 32
        self.uniform_3 = conv_layer(256, 32, with_maxpool= False)  # 16 x 16 x 8 x 32

        self.values_linear = nn.Linear(32, 32)  # 16 x 16 x 8 x 32

        # norm layers
        self.norm64 = nn.BatchNorm3d(64, track_running_stats=False)
        self.norm128 = nn.BatchNorm3d(128, track_running_stats=False)
        self.norm256 = nn.BatchNorm3d(256, track_running_stats=False)
        self.norm32 = nn.BatchNorm3d(32, track_running_stats=False)

        # embed dim 
        self.attn_head = nn.MultiheadAttention(32, num_heads = 4)

        # fully connected layers
        self.linear1 = nn.Linear(256, 1)
        self.linear2 = nn.Linear(16*16*8, 1)
        # relu activation
        self.relu = nn.ReLU()

    def forward(self, x):
        l_out1 = self.norm64(self.layer_1(x))
        l_out2 = self.norm128(self.layer_2(l_out1))
        l_out3 = self.norm256(self.layer_3(l_out2))

        key1 = self.uniform_1(l_out1)
        key2 = self.uniform_2(l_out2)
        key3 = self.uniform_3(l_out3)

        num_samples = x.shape[0]
        
        pdb.set_trace()
        
        keys = torch.cat([
                #    key1.view(num_samples, 32, -1), 
                #    key2.view(num_samples, 32, -1),
                   key3.view(num_samples, 32, -1)], dim = -1)
        
        query = self.uniform_0(x).view(num_samples, 32, -1)
        
        values = self.relu(self.values_linear(keys.transpose(1, 2)))
        pdb.set_trace()
        attn_output = self.attn_head(query.permute(2, 0, 1), keys.permute(2, 0, 1), values.permute(1, 0, 2))
        
        x = x.view(num_samples, 256, -1).transpose(1, 2)
        x = self.relu(self.linear1(x).squeeze(-1))
        x = self.linear2(x)
        return x

        
if __name__ == '__main__':
    data = torch.randn(6, 1, 128, 128, 64).to(device)
    model = Unet3d().to(device)
    out = model(data)
    print(out.shape)
    exit()

    model = NonLocalSE().to(device)
    out = model(data)

    # layer = conv_layer(1, 32)
    model = CNN3d()
    out = model(data)
    # print(out.shape)
