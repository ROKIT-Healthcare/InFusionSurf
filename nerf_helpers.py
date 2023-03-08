import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tinycudann as tcnn
import grid
from torch.utils.cpp_extension import load

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2mae = lambda x, y : torch.mean(torch.abs((x - y)))
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
to_depth16 = lambda x: (1000 * x).astype(np.uint16)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        
        if self.kwargs['gaussian']:
            B = torch.randn(3, 256) * 10
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                if self.kwargs['gaussian']:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq, B=B: p_fn(torch.matmul(x, B) * freq))
                else:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder_obj(multires):
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
                'gaussian': False,
    }

    return Embedder(**embed_kwargs)

def get_hashgrid_embedder_obj(input_ch, multires):
    config = {
        'otype': 'HashGrid',
        'n_levels': 2 * multires,
        'n_features_per_level': 2,
        'log2_hashmap_size': 15,
        'base_resolution': 16,
        'per_level_scale': 1.5
    }
    
    encoder = tcnn.Encoding(input_ch, config, dtype=torch.float32)
    
    return encoder

def get_embedder(multires, i=0):
    trainable = False
    
    if i == -1:
        return nn.Identity(), 3, trainable
    
    if i == 0:
        embedder_obj = get_embedder_obj(multires)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        out_dim = embedder_obj.out_dim
    elif i == 1:
        embed = get_hashgrid_embedder_obj(3, multires)
        out_dim = embed.n_output_dims
        trainable = True
    else:
        embedder_obj = get_embedder_obj(multires)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        out_dim = embedder_obj.out_dim
    
    return embed, out_dim, trainable


'''Model'''
class FastSurf(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 voxel_size=0.1, feature_dim=12,
                 D=3, W=128,
                 viewbase_pe=4, i_view_embed=0,
                 n_frame_features=2
        ):
        super(FastSurf, self).__init__()
        
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        
        # determine init grid resolution
        self._set_grid_resolution(voxel_size)
            
        self.n_frame_features = n_frame_features
        
        self.feature_dim = feature_dim
        self.feature = grid.create_grid(
            'DenseGrid', channels=self.feature_dim, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
            config={}
        )
        
        view_input_ch = 0
        self.view_embed_fn, view_input_ch, _ = get_embedder(viewbase_pe, i_view_embed)

        dim0 = view_input_ch + n_frame_features
        dim0 += self.feature_dim
        
        self.densitynet = nn.Sequential(
            nn.Linear(self.feature_dim, W), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True))
                for _ in range(D - 2)
            ],
            nn.Linear(W, 1),
        )
        nn.init.constant_(self.densitynet[-1].bias, 0)
        
        self.rgbnet = nn.Sequential(
            nn.Linear(dim0, W), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True))
                for _ in range(D - 2)
            ],
            nn.Linear(W, 3),
        )
        nn.init.constant_(self.rgbnet[-1].bias, 0)        

    def _set_grid_resolution(self, voxel_size):
        # Determine grid resolution
        self.register_buffer('voxel_size', torch.Tensor([voxel_size]))
        
        self.world_size = torch.ceil((self.xyz_max - self.xyz_min) / voxel_size).long()
        
        print('fastsurf: voxel_size      ', self.voxel_size[0])
        print('fastsurf: world_size      ', self.world_size)

    @torch.no_grad()
    def scale_volume_grid(self, voxel_size):
        print('fastsurf: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(voxel_size)
        print('fastsurf: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.feature.scale_volume_grid(self.world_size)

        print('fastsurf: scale_volume_grid finish')

    def forward(self, x, sdf_only=False):
        if sdf_only:
            ray_pts = x
        elif self.n_frame_features > 0:
            ray_pts, frame_features, viewdirs = torch.split(x, [3, self.n_frame_features, 3], dim=-1)
        else:
            ray_pts, viewdirs = torch.split(x, [3, 3], dim=-1)
        
        assert len(ray_pts.shape)==2 and ray_pts.shape[-1]==3, 'Only suport point queries in [N, 3] format'

        ret_dict = {}
        N = len(ray_pts)
        
        # query for color        
        feature = self.feature(ray_pts)
        
        sdf = self.densitynet(feature)
        
        if sdf_only:
            return sdf

        viewdirs_emb = self.view_embed_fn(viewdirs)

        if self.n_frame_features > 0:
            rgb_feat = torch.cat([feature, viewdirs_emb, frame_features], -1)
        else:
            rgb_feat = torch.cat([feature, viewdirs_emb], -1)
            
        rgb = self.rgbnet(rgb_feat)
            
        outputs = torch.cat([rgb, sdf], -1)
        
        return outputs
    
# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i+0.5-W*.5)/focal, -(j + 0.5 - H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i + 0.5 - W*.5)/focal, -(j + 0.5 - H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def get_camera_rays_np(H, W, focal):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i + 0.5 - W*.5)/focal, -(j + 0.5 - H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = dirs
    return rays_d


def get_rays_np_random(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')

    i_rand = np.random.rand(*i.shape)
    j_rand = np.random.rand(*j.shape)

    dirs = np.stack([(i + i_rand - W*.5)/focal, -(j + j_rand - H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples