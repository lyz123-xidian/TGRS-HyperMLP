import torch
import numpy as np
from torch import nn
import math
from einops.layers.torch import Rearrange
from einops import rearrange
# from rpca import robust_pca
def shrink(X,tau):
    V = torch.clone(X).reshape((X.numel(),))
    for i in range(V.numel()):
        t = torch.abs(V[i]) - tau
        a = torch.maximum(t, torch.tensor(0))
        # a = torch.max(t, 0)[0]
        V[i] = torch.copysign(a, V[i])
        # if V[i] == -0:
        #     V[i] = 0

    c = X.shape
    pp = V.reshape(X.shape)
    return pp
def svd_shrink(X, tau):
    U,s,V = torch.linalg.svd(X,full_matrices=False)
    a1 = torch.diag(shrink(s, tau))
    # b = a1 / V
    b = torch.mm(a1, V)
    a = torch.mm(U, b)

    return a

def frobeniusNorm(X):
    """
    Evaluate the Frobenius norm of X
    Returns sqrt(sum_i sum_j X[i,j] ^ 2)
    """
    accum = torch.tensor(0)
    V = torch.reshape(X, (X.numel(),))
    for i in range(V.numel()):
        b = torch.abs(V[i] ** 2)
        accum = accum + b
        # accum += torch.abs(V[i] ** 2)
    return torch.sqrt(accum)

def converged(M, L, S):
    """
    A simple test of convergence based on accuracy of matrix reconstruction
    from sparse and low rank parts
    """
    a = frobeniusNorm(M - L - S)
    b = frobeniusNorm(M)
    error = torch.div(a, b)
    # print("error =", error)
    return error <= 10e-6

def L1Norm(X):
    """
    Evaluate the L1 norm of X
    Returns the max over the sum of each column of X
    """
    return torch.max(torch.sum(X, dim=0))

def robust_pca(M):
    """
    Decompose a matrix into low rank and sparse components.
    Computes the RPCA decomposition using Alternating Lagrangian Multipliers.
    Returns L,S the low rank and sparse components respectively
    """
    L = torch.zeros(M.shape)
    L = L.cuda()
    S = torch.zeros(M.shape)
    S = S.cuda()
    Y = torch.zeros(M.shape)
    Y = Y.cuda()
    # print(M.shape)
    mu = torch.div(M.shape[0] * M.shape[1], 4.0 * L1Norm(M))
    # mu = (M.shape[0] * M.shape[1]) / (4.0 * L1Norm(M))
    lamb = max(M.shape) ** -0.5
    tt = mu * lamb
    while not converged(M, L, S):
        L = svd_shrink(M - S - (mu ** -1) * Y, mu)
        S = shrink(M - L + (mu ** -1) * Y, tt)
        Y = Y + mu * (M - L - S)
    return L, S


class spatialLayer(nn.Module):
    def __init__(self,inchan,channel,reduction):
        super(spatialLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.mlp = nn.Sequential(
            # nn.Conv1d(inchan, channel//reduction,1,bias=False),
            nn.Conv1d(inchan, channel, 1, bias=False),
            # nn.BatchNorm1d(channel),
            # nn.BatchNorm1d(channel//reduction),
            nn.ReLU(inplace=True),
            # nn.Conv1d(channel//reduction, inchan, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        spa_out = self.mlp(self.max_pool(x))
        spa_weight = self.sigmoid(spa_out)
        return spa_weight

class channelLayer(nn.Module):
    def __init__(self,inchan, channel, reduction):
        super(channelLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Conv1d(inchan, inchan, 1, bias=False),
            # nn.BatchNorm1d(inchan),
            # nn.Conv1d(inchan, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            # nn.Conv1d(channel // reduction, inchan, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        channel_out = self.mlp(self.avg_pool(x))
        channel_weight = self.sigmoid(channel_out)
        return channel_weight
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, num_patch_lidar,dropout = 0.):
        super().__init__()
        self.spa_num = num_patch
        self.ln = nn.LayerNorm(dim)
        self.re1 = Rearrange('b n d -> b d n')
        self.ff1 = FeedForward(num_patch, token_dim, dropout)
        self.re2 = Rearrange('b d n -> b n d')
        self.conv1 = nn.Conv1d(dim, 196, kernel_size=1)
        self.conv2 = nn.Conv1d(196,dim, kernel_size=1)
        # self.conv3 = nn.Conv1d(512,dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(dim)
        self.pacth_float = int(dim)
        self.pacth_float1 = int(dim/2)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        # self.spatialweight = spatialLayer(dim,self.spa_num,reduction=12)
        # self.channelweight = channelLayer(self.spa_num,dim,reduction=16)
        self.spatialweight = spatialLayer(self.spa_num, dim, reduction=11)
        self.channelweight = channelLayer(dim,self.spa_num,reduction=16)
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.token_mix_patch = nn.Sequential(
            nn.LayerNorm(16),
            Rearrange('b n d -> b d n'),
            FeedForward(121, 64, dropout),
            Rearrange('b d n -> b n d')
        )
        self.token_mix_all = nn.Sequential(
            nn.LayerNorm(176),
            Rearrange('b n d -> b d n'),
            FeedForward(88, 128, dropout),
            Rearrange('b d n -> b n d')
        )
        self.band_wise = nn.Sequential(
            nn.Conv1d(in_channels=num_patch, out_channels=num_patch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_patch),
            nn.ReLU(),
        )
        self.channel_mix1 = nn.Sequential(
            nn.LayerNorm(self.pacth_float),
            FeedForward(self.pacth_float, channel_dim, dropout),
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )
        self.channel_mix_patch = nn.Sequential(
            nn.LayerNorm(16),
            FeedForward(16, 128, dropout),
        )
        self.channel_mix_all = nn.Sequential(
            nn.LayerNorm(176),
            FeedForward(176, 128, dropout),
        )
        self.lidar_channelmix = nn.Sequential(
            nn.LayerNorm(self.pacth_float),
            FeedForward(self.pacth_float, int(channel_dim), dropout),)
        self.lidar_tokenmix = nn.Sequential(
            nn.LayerNorm(self.pacth_float),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch_lidar, int(token_dim), dropout),
            Rearrange('b d n -> b n d')
        )
        self.lidar_tokenmix2 = nn.Sequential(
            nn.LayerNorm(self.pacth_float1),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch_lidar*2, int(token_dim), dropout),
            Rearrange('b d n -> b n d')
        )
        self.conv_fea_exa = nn.Sequential(
            nn.Conv1d(in_channels=121, out_channels=121, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(121),
            nn.ReLU()
        )
        self.te_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.affi_conv = nn.Conv1d(in_channels=dim * 2, out_channels=dim, kernel_size=1)
        self.turn = Rearrange('c h w  -> w h c')
        # self.turn = Rearrange('h w  -> w h')
        self.turn_back = Rearrange('c h w  -> w h c')
        # self.turn_back = Rearrange('h w  -> w h')
        self.d2_affi = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.d2_affi_two = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)

    def forward(self, x, x1):
        x_up = x1[:,:,:64]
        x_down = x1[:,:,64:]
        x_cat = torch.cat([x_up, x_down], dim=1)
        x_out = self.lidar_tokenmix2(x_cat)
        y1 = x_out[:,:225,:]
        y2 = x_out[:,225:,:]
        y = torch.cat([y1, y2], dim=2)
        # # band_wise = self.band_wise(x)
        x1 = x1 + y
        # x1 = x1+self.lidar_tokenmix(x1)
        x1 = x1+self.lidar_channelmix(x1)

        # band_wise = self.band_wise(x)
        # x = x + self.channel_mix(x)
        # b,_,_, = x.shape
        # x = x.reshape(b,121,8,16)
        # x = rearrange(x, 'b c h w -> (b h) c w')
        # x = rearrange(x, 'b c h w -> (b c) h w')
        # x = x.reshape(5*88, 11, 16)
        # p = self.token_mix_patch(x)
        # x = x + self.token_mix_patch(x)
        # x = x + self.channel_mix_patch(x)
        # # x = self.conv_fea_exa(x)
        # x = x.reshape(b,121,128)


        # #jiaocha affi
        # x = self.turn(x)
        # x1 = self.turn(x1)
        # x = self.d2_affi(x)
        # x1 = self.d2_affi(x1)
        # # x = self.turn_back(x)
        # # x1 = self.turn_back(x1)
        # affi = torch.cat((x,x1),dim=1)
        # affi_matrix = self.d2_affi_two(affi)
        # affi_matrix = self.turn_back(affi_matrix)
        # affi_matrix = affi_matrix.mean(dim=1)
        # affi_matrix = torch.sigmoid(affi_matrix)
        # affi_matrix1 = affi_matrix.repeat(121,1,1)
        # affi_matrix2 = affi_matrix.repeat(225,1,1)
        # affi_matrix1 = rearrange(affi_matrix1, 'w h c -> h w c')
        # affi_matrix2 = rearrange(affi_matrix2, 'w h c -> h w c')
        # x = self.turn_back(x)
        # x1 = self.turn_back(x1)
        # x = torch.mul(x,affi_matrix1)
        # x1 = torch.mul(x1, affi_matrix2)
        # x = self.conv_fea_exa(x)
        # x = x + self.token_mix(x)
        # x = x + self.channel_mix(x)

        # x_token = self.ln(x)
        # x_token = self.re1(x_token)

        ### get weight
        # spa_weight = self.channelweight(x_token)
        # ### normalization
        # meana = torch.max(spa_weight,dim=1,keepdim=True).values
        # mina = torch.min(spa_weight,dim=1,keepdim=True).values
        # temp = meana - mina
        # spa_weight11 = torch.sub(spa_weight,mina)
        # spa_weight12 = torch.div(spa_weight11,temp)
        #### entropy
        # a = -(torch.log(spa_weight))
        # a = torch.from_numpy(-np.log2(a))
        # spa_weight1 = torch.mul(a,spa_weight12)
        # spa_weight2 = 1 - a
        ### fead forward
        # x_token = self.ff1(x_token)
        # x_token = torch.mul(x_token,spa_weight2)
        # x_token = self.re2(x_token)
        # x = x + x_token

        # x_out = x + x1
        # x1 = x[:,:,:64]
        # x2 = x[:,:,64:]
        # x_cat = torch.cat([x1, x2], dim=1)
        # x_out = self.channel_mix1(x_cat)
        # y1 = x_out[:,:121,:]
        # y2 = x_out[:,121:,:]
        # y = torch.cat([y1, y2], dim=2)
        # # x = x + self.token_mix(x)
        # # band_wise = self.band_wise(x)
        # x = x + y
        # x = x + self.channel_mix(x)

        return x, x1


class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim,lidar_size):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2
        self.lidar_dim = int(dim)
        self.num_patch_lidar = (lidar_size//patch_size) ** 2
        self.num_patch22 = in_channels
        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv2d(in_channels, dim, patch_size, patch_size),
        #     Rearrange('b c h w -> b (h w) c'),
        # )
        self.to_patch_embedding_lidar = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.Conv1d(1, self.lidar_dim, patch_size, patch_size),
            Rearrange('b c h -> b h c'),
        )
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.Conv1d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h -> b h c'),
        )


        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim,self.num_patch_lidar))

        self.layer_norm = nn.LayerNorm(dim)
        self.layer_norm_patch = nn.LayerNorm(176)
        self.layer_norm_lidar = nn.LayerNorm(self.lidar_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )
        self.mlp_head_lidar = nn.Sequential(
            nn.Linear(self.lidar_dim, num_classes)
        )
        self.conv_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Linear(dim, num_classes)
            # nn.Linear(dim*2,int(dim/2)),
            # nn.Linear(int(dim/2),num_classes)
        )
        self.conv_out_patch = nn.Sequential(
            # nn.Linear(304, dim),
            # nn.Linear(dim, int(dim / 2)),
            nn.Linear(304, num_classes)
        )
        self.te_conv = nn.Conv1d(in_channels=dim,out_channels=dim,kernel_size=1)
        self.affi_conv = nn.Conv1d(in_channels=dim*2, out_channels=dim*2, kernel_size=1)
        # self.turn = Rearrange('c h w  -> w h c')
        self.turn = Rearrange('h w  -> w h')
        # self.turn_back = Rearrange('c h w  -> w h c')
        self.turn_back = Rearrange('h w  -> w h')
        self.d2_affi = nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=1)
        self.d2_affi_two = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)

    def forward(self, x, x1):


        x = self.to_patch_embedding(x)
        x1 = self.to_patch_embedding_lidar(x1)
        x_center = x[:,61,:]
        # x_patch = x.reshape(5*88,11,16)#change with input
        for mixer_block in self.mixer_blocks:
            x, x1 = mixer_block(x, x1)
        x1 = self.layer_norm_lidar(x1)
        # x = self.layer_norm(x)
        x = self.layer_norm(x)
        #xian affi zai mean
        # x_nmf = torch.cat((x,x1),dim=1)
        # x_1 = self.turn(x1)
        # x_2 = self.turn(x)
        # x_1 = self.d2_affi(x_1)
        # x_2 = self.d2_affi(x_2)
        # x_affi = torch.cat((x_1,x_2),dim=1)
        # x_affi = self.d2_affi_two(x_affi)
        # x_affi = self.turn_back(x_affi)
        # x_all = torch.cat((x, x1), dim=1)
        # x_all_1 = torch.mul(x_all, x_affi)
        #
        # x_out = x_all_1.mean(dim=1)
        x1 = x1.mean(dim=1)
        x = x.mean(dim=1)
        # x_all = torch.cat((x,x1),dim=1)
        #
        #xian mean zai affi

        # x_aff1 = self.turn(x1)
        # x_aff2 = self.turn(x)
        # affinity1 = self.te_conv(x_aff1)
        # affinity2 = self.te_conv(x_aff2)
        # aaa1 = torch.cat((affinity1,affinity2), dim=0)
        # affinity_matrix = self.affi_conv(aaa1)
        # # rpca data conver
        # # x_rpca = robust_pca(x_all)[0]
        # ###
        # affinity_matrix = self.turn_back(affinity_matrix)
        # x_all_1 = torch.mul(x_all, affinity_matrix)
        x_out = self.conv_out(x1)
        # x_out = self.conv_out(x_all)
        # x = self.mlp_head(x)
        # x1 = self.mlp_head_lidar(x1)
        # x_out = x + x1
        # return self.mlp_head(x)
        return x_out




if __name__ == "__main__":
    img = torch.ones([12, 63, 11, 11])
    img2 = torch.randn([12, 1, 15, 15])
    model = MLPMixer(in_channels=63, image_size=11, patch_size=1, num_classes=6,
                     dim=128, depth=2, token_dim=128, channel_dim=128,lidar_size=15)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img,img2)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]




