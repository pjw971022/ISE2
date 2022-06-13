from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from perceiver.pos_encoding import build_position_encoding
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4, base = 2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(0., log(max_freq / 2) / log(base), num_bands, base = base, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)
    
class PostNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = 'geglu', more_dropout = False, xavier_init = False):
        super().__init__()
        act_in_dim = dim * mult
        act_out_dim = act_in_dim
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'geglu':
            act_in_dim *= 2
            self.activation = GEGLU()
        else:
            raise NotImplementedError("Invalid activation function")
            
        self.net = nn.Sequential(
            nn.Linear(dim, act_in_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(act_out_dim, dim),
            nn.Dropout(dropout) if more_dropout else nn.Identity()
        )
        
        if xavier_init:
            self._reset_parameter()
    
    def _reset_parameter(self):
        def fn(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        self.net.apply(fn)

    def forward(self, x):
        return self.net(x)

class ThinFeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = 'geglu', more_dropout = False, xavier_init = False):
        super().__init__()
        act_in_dim = dim * mult
        act_out_dim = act_in_dim
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError("Invalid activation function")
            
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            self.activation,
            nn.Dropout(dropout)
        )
        
        if xavier_init:
            self._reset_parameter()
    
    def _reset_parameter(self):
        def fn(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        self.net.apply(fn)

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0., attn_type = 'transformer', more_dropout = False, xavier_init = False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.attn_holder = nn.Identity()
        
        self.attn_type = attn_type
        self.attn_matrix_dropout = nn.Dropout(dropout) if more_dropout else nn.Identity()
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        if xavier_init:
            self._reset_parameter()
        
    def _reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        # nn.init.constant_(self.to_q.bias, 0.0)
        # nn.init.constant_(self.to_k.bias, 0.0)
        # nn.init.constant_(self.to_v.bias, 0.0)

    def forward(self, x, context = None, mask = None, k_pos = None, q_pos = None):
        h = self.heads

        q = self.to_q(x if q_pos is None else x + q_pos)
        context = default(context, x)
        k = self.to_k(context if k_pos is None else context + k_pos)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(mask, max_neg_value)

        # attention, what we cannot get enough of
        if self.attn_type == 'transformer':
            attn = sim.softmax(dim = -1)
        elif self.attn_type == 'slot':
            attn = sim.softmax(dim = 1)
            attn = attn / (attn.sum(dim = -1, keepdim = True) + 1e-7)
        else:
            raise NotImplementedError
        
        if torch.isnan(attn).any():
            import ipdb; ipdb.set_trace()
            
        attn = self.attn_holder(attn)
        attn = self.attn_matrix_dropout(attn)
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class
class DETR(nn.Module):
    def __init__(
        self,
        *,
        depth,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        self_before_cross_attn = 0,
        query_self_attn = 1,
        pos_enc_type = 'none',
        last_fc = True,
        post_norm = True, 
        activation = 'geglu',
        last_ln = False,
        ff_mult = 4,
        cross_attn_type = 'transformer'
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
            num_freq_bands: Number of freq bands, with original value (2 * K + 1)
            depth: Depth of net.
            max_freq: Maximum frequency, hyperparameter depending on how
                fine the data is.
            freq_base: Base for the frequency
            input_channels: Number of channels for each token of the input.
            input_axis: Number of axes for input data (2 for images, 3 for video)
            num_latents: Number of latents, or induced set points, or centroids.
                Different papers giving it different names.
            latent_dim: Latent dimension.
            cross_heads: Number of heads for cross attention. Paper said 1.
            latent_heads: Number of heads for latent self attention, 8.
            cross_dim_head: Number of dimensions per cross attention head.
            latent_dim_head: Number of dimensions per latent self attention head.
            num_classes: Output number of classes.
            attn_dropout: Attention dropout
            ff_dropout: Feedforward dropout
            weight_tie_layers: Whether to weight tie layers (optional).
            fourier_encode_data: Whether to auto-fourier encode the data, using
                the input_axis given. defaults to True, but can be turned off
                if you are fourier encoding the data yourself.
            self_per_cross_attn: Number of self attention blocks per cross attn.
        """
        super().__init__()
        self.input_axis = input_axis
        self.num_classes = num_classes

        input_dim = input_channels
        self.input_dim = input_channels
        self.pos_enc = build_position_encoding(input_dim, pos_enc_type, self.input_axis)

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        
        self.prenorm = PreNorm if not post_norm else lambda dim, fn, context_dim=None: fn
        self.postnorm = PostNorm if post_norm else nn.Identity
        
        # * decoder cross attention layers
        get_cross_attn = \
            lambda: self.prenorm(
                latent_dim, 
                Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout, attn_type = cross_attn_type), 
                context_dim = input_dim)
        get_cross_ff = lambda: self.prenorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout, activation = activation, mult=ff_mult))
        get_cross_postnorm = lambda: self.postnorm(latent_dim)
        
        # * self attention of queries (first self attention layer of decoder)
        get_latent_attn = lambda: self.prenorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: self.prenorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout, activation = activation, mult=ff_mult))
        get_latent_postnorm = lambda: self.postnorm(latent_dim)
        
        # * encoder layers
        # FIXME add option to encoder layers to have its own hyper-parameter option, not just following latent layer options
        get_pre_self_attn = lambda: self.prenorm(input_dim, Attention(input_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_pre_self_ff = lambda: self.prenorm(input_dim, FeedForward(input_dim, dropout = ff_dropout, activation = activation, mult=ff_mult))
        get_pre_self_postnorm = lambda: self.postnorm(input_dim)

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff, get_pre_self_attn, get_pre_self_ff = map(cache_fn, \
            (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff, get_pre_self_attn, get_pre_self_ff)
        )
        
        # self attention before going into decoder, coresponding to the DETR encoder
        self.pre_self_attns = nn.ModuleList([])
        for _ in range(self_before_cross_attn):
            self.pre_self_attns.append(nn.ModuleList([
                get_pre_self_attn(**{'_cache': False}),
                get_pre_self_postnorm(),
                get_pre_self_ff(**{'_cache': False}),
                get_pre_self_postnorm()
            ]))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_postnorm(),
                get_cross_attn(**cache_args),
                get_cross_postnorm(),
                get_cross_ff(**cache_args),
                get_cross_postnorm()
            ]))

        # Last FC layer
        if not last_fc:
            assert latent_dim == self.num_classes
        self.last_layer = nn.Sequential(
            nn.LayerNorm(latent_dim) if last_ln and not post_norm else nn.Identity(),
            nn.Linear(latent_dim, num_classes) if last_fc else nn.Identity()
        )
    
    def null_pos_enc(self):
        self.pos_enc = build_position_encoding(self.input_dim, 'none', self.input_axis)

    def forward(self, data, mask = None):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # concat to channels of data and flatten axis
        pos = self.pos_enc(data)
        
        data = rearrange(data, 'b ... d -> b (...) d')
        
        elem_q_pos = repeat(self.latents, 'n d -> b n d', b = b)
        tgt = torch.zeros_like(elem_q_pos)
        
        # layers
        for pre_self_attn, pn1, self_ff, pn2 in self.pre_self_attns:
            data = pre_self_attn(data, mask = mask, q_pos = pos, k_pos = pos) + data
            data = pn1(data)
            data = self_ff(data) + data
            data = pn2(data)

        for latent_attn, pn1, cross_attn, pn2, cross_ff, pn3 in self.layers:
            tgt = latent_attn(tgt, q_pos = elem_q_pos, k_pos = elem_q_pos) + tgt
            tgt = pn1(tgt)
            tgt = cross_attn(tgt, context = data, mask = mask, q_pos = elem_q_pos, k_pos = pos) + tgt
            tgt = pn2(tgt)
            tgt = cross_ff(tgt) + tgt
            tgt = pn3(tgt)
        
        # last_attn_map = rearrange(last_attn_map, '(b h) n d -> b h n d', h = 8).mean(dim=1)
        return self.last_layer(tgt)


class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        depth,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        self_before_cross_attn = 0,
        query_self_attn = 1,
        pos_enc_type = 'none',
        last_fc = True,
        pre_norm = True,
        post_norm = True, 
        activation = 'geglu',
        last_ln = False,
        ff_mult = 4,
        cross_attn_type = 'transformer',
        more_dropout = False,
        xavier_init = False,
        thin_ff = False,
        query_fixed = False,
        query_xavier_init = False,
        query_type = 'learned'
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
            num_freq_bands: Number of freq bands, with original value (2 * K + 1)
            depth: Depth of net.
            max_freq: Maximum frequency, hyperparameter depending on how
                fine the data is.
            freq_base: Base for the frequency
            input_channels: Number of channels for each token of the input.
            input_axis: Number of axes for input data (2 for images, 3 for video)
            num_latents: Number of latents, or induced set points, or centroids.
                Different papers giving it different names.
            latent_dim: Latent dimension.
            cross_heads: Number of heads for cross attention. Paper said 1.
            latent_heads: Number of heads for latent self attention, 8.
            cross_dim_head: Number of dimensions per cross attention head.
            latent_dim_head: Number of dimensions per latent self attention head.
            num_classes: Output number of classes.
            attn_dropout: Attention dropout
            ff_dropout: Feedforward dropout
            weight_tie_layers: Whether to weight tie layers (optional).
            fourier_encode_data: Whether to auto-fourier encode the data, using
                the input_axis given. defaults to True, but can be turned off
                if you are fourier encoding the data yourself.
            self_per_cross_attn: Number of self attention blocks per cross attn.
        """
        super().__init__()
        self.input_axis = input_axis
        self.num_classes = num_classes

        input_dim = input_channels
        self.input_dim = input_channels
        self.pos_enc = build_position_encoding(input_dim, pos_enc_type, self.input_axis)

        self.num_latents = num_latents
        self.query_type = query_type
        self.latent_dim = latent_dim
        
        if self.query_type == 'learned':
            self.latents = nn.Parameter(torch.randn(self.num_latents, latent_dim))
            if query_fixed:
                self.latents.requires_grad = False
            if query_xavier_init:
                nn.init.xavier_normal_(self.latents)
        elif self.query_type == 'slot':
            self.slots_mu = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, latent_dim)), gain=nn.init.calculate_gain("linear"))
            self.slots_log_sigma = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, latent_dim)), gain=nn.init.calculate_gain("linear"))
        else:
            raise NotImplementedError
        
        assert (pre_norm or post_norm)
        self.prenorm = PreNorm if pre_norm else lambda dim, fn, context_dim=None: fn
        self.postnorm = PostNorm if post_norm else nn.Identity
        
        ff = ThinFeedForward if thin_ff else FeedForward
        
        # * decoder cross attention layers
        get_cross_attn = \
            lambda: self.prenorm(
                latent_dim, 
                Attention(
                    latent_dim, input_dim,
                    heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout, attn_type = cross_attn_type, more_dropout = more_dropout, xavier_init = xavier_init
                ), 
                context_dim = input_dim)
        get_cross_ff = lambda: self.prenorm(latent_dim, ff(latent_dim, dropout = ff_dropout, activation = activation, mult=ff_mult, more_dropout = more_dropout, xavier_init = xavier_init))
        get_cross_postnorm = lambda: self.postnorm(latent_dim)
        
        # * self attention of queries (first self attention layer of decoder)
        get_latent_attn = lambda: self.prenorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout, more_dropout = more_dropout, xavier_init = xavier_init))
        get_latent_ff = lambda: self.prenorm(latent_dim, ff(latent_dim, dropout = ff_dropout, activation = activation, mult=ff_mult, more_dropout = more_dropout, xavier_init = xavier_init))
        get_latent_postnorm = lambda: self.postnorm(latent_dim)
        
        # * encoder layers
        # FIXME add option to encoder layers to have its own hyper-parameter option, not just following latent layer options
        get_pre_self_attn = lambda: self.prenorm(input_dim, Attention(input_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout, more_dropout = more_dropout, xavier_init = xavier_init))
        get_pre_self_ff = lambda: self.prenorm(input_dim, ff(input_dim, dropout = ff_dropout, activation = activation, mult=ff_mult, more_dropout = more_dropout, xavier_init = xavier_init))
        get_pre_self_postnorm = lambda: self.postnorm(input_dim)

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff, get_pre_self_attn, get_pre_self_ff = map(cache_fn, \
            (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff, get_pre_self_attn, get_pre_self_ff)
        )

        self.layers = nn.ModuleList([])
        
        # self attention before going into decoder, coresponding to the DETR encoder
        self.pre_self_attns = nn.ModuleList([])
        for _ in range(self_before_cross_attn):
            self.pre_self_attns.append(nn.ModuleList([
                get_pre_self_attn(**{'_cache': False}),
                get_pre_self_postnorm(),
                get_pre_self_ff(**{'_cache': False}),
                get_pre_self_postnorm()
            ]))
        
        # self attention for decoder query (not necessary but following DETR's choice)
        self.query_self_attns = nn.ModuleList([])
        for _ in range(query_self_attn):
            self.query_self_attns.append(nn.ModuleList([
                get_latent_attn(**{'_cache': False}),
                get_latent_postnorm(),
                get_latent_ff(**{'_cache': False}),
                get_latent_postnorm()
            ]))

        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}
            
            # self attention after cross attention, only exists in perceiver arch.
            self_attns = nn.ModuleList([])
            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_postnorm(),
                    get_latent_ff(**cache_args),
                    get_latent_postnorm()
                ]))
            
            # cross attention layer, DETR decoder
            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_postnorm(),
                get_cross_ff(**cache_args),
                get_cross_postnorm(),
                self_attns
            ]))

        # Last FC layer
        if not last_fc:
            assert latent_dim == self.num_classes
        self.last_layer = nn.Sequential(
            nn.LayerNorm(latent_dim) if last_ln and not post_norm else nn.Identity(),
            nn.Linear(latent_dim, num_classes) if last_fc else nn.Identity()
        )
        
        self.encoder_output_holder = nn.Identity()
        self.decoder_output_holder = nn.Identity()
        
    def get_queries(self, b):
        if self.query_type == 'learned':
            ret = repeat(self.latents, 'n d -> b n d', b = b)
        elif self.query_type == 'slot':
            slots_init = torch.randn((b, self.num_latents, self.latent_dim)).cuda()
            ret = self.slots_mu + self.slots_log_sigma.exp() * slots_init
        return ret

    def forward(self, data, mask = None):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # concat to channels of data and flatten axis
        pos = self.pos_enc(data)
        
        data = rearrange(data, 'b ... d -> b (...) d')
        
        x = self.get_queries(b).type_as(data)
        # layers
        for pre_self_attn, pn1, self_ff, pn2 in self.pre_self_attns:
            data = pre_self_attn(data, mask = mask, q_pos = pos, k_pos = pos) + data
            data = pn1(data)
            data = self_ff(data) + data
            data = pn2(data)
            
        data = self.encoder_output_holder(data)
        last_attn_map = None
        for query_self_attn, pn1, self_ff, pn2 in self.query_self_attns:
            x = query_self_attn(x) + x
            x = pn1(x)
            x = self_ff(x) + x
            x = pn2(x)

        for cross_attn, pn1, cross_ff, pn2, self_attns in self.layers:
            last_attn_map = cross_attn(x, context = data, mask = mask, k_pos = pos, q_pos = None)
            
            x = last_attn_map + x
            x = pn1(x)
            x = cross_ff(x) + x
            x = pn2(x)
            # only for perceiver arch. not used in current implementation
            for self_attn, pn1, self_ff, pn2 in self_attns:
                x = self_attn(x) + x
                x = pn1(x)
                x = self_ff(x) + x
                x = pn2(x)
                last_attn_map = self_attn(x)
        x = self.decoder_output_holder(x)
        # print('lam shape1: ',last_attn_map.shape)
        # last_attn_map = rearrange(last_attn_map, '(b h) n d -> b h n d', h = 8).mean(dim=1)
        # print('lam shape2: ',last_attn_map.shape)
        
        return self.last_layer(x), last_attn_map
    
class PerceiverEncRes(nn.Module):
    def __init__(
        self,
        *,
        depth,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        self_before_cross_attn = 0,
        query_self_attn = 1,
        pos_enc_type = 'none',
        last_fc = True,
        pre_norm = True,
        post_norm = True, 
        activation = 'geglu',
        last_ln = False,
        ff_mult = 4,
        cross_attn_type = 'transformer',
        more_dropout = False,
        xavier_init = False,
        thin_ff = False,
        query_fixed = False,
        query_xavier_init = False,
        query_type = 'learned'
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
            num_freq_bands: Number of freq bands, with original value (2 * K + 1)
            depth: Depth of net.
            max_freq: Maximum frequency, hyperparameter depending on how
                fine the data is.
            freq_base: Base for the frequency
            input_channels: Number of channels for each token of the input.
            input_axis: Number of axes for input data (2 for images, 3 for video)
            num_latents: Number of latents, or induced set points, or centroids.
                Different papers giving it different names.
            latent_dim: Latent dimension.
            cross_heads: Number of heads for cross attention. Paper said 1.
            latent_heads: Number of heads for latent self attention, 8.
            cross_dim_head: Number of dimensions per cross attention head.
            latent_dim_head: Number of dimensions per latent self attention head.
            num_classes: Output number of classes.
            attn_dropout: Attention dropout
            ff_dropout: Feedforward dropout
            weight_tie_layers: Whether to weight tie layers (optional).
            fourier_encode_data: Whether to auto-fourier encode the data, using
                the input_axis given. defaults to True, but can be turned off
                if you are fourier encoding the data yourself.
            self_per_cross_attn: Number of self attention blocks per cross attn.
        """
        super().__init__()
        self.input_axis = input_axis
        self.num_classes = num_classes

        input_dim = input_channels
        self.input_dim = input_channels
        self.pos_enc = build_position_encoding(input_dim, pos_enc_type, self.input_axis)

        self.num_latents = num_latents
        self.query_type = query_type
        self.latent_dim = latent_dim
        
        if self.query_type == 'learned':
            self.latents = nn.Parameter(torch.randn(self.num_latents, latent_dim))
            if query_fixed:
                self.latents.requires_grad = False
            if query_xavier_init:
                nn.init.xavier_normal_(self.latents)
        elif self.query_type == 'slot':
            self.slots_mu = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, latent_dim)), gain=nn.init.calculate_gain("linear"))
            self.slots_log_sigma = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, latent_dim)), gain=nn.init.calculate_gain("linear"))
        else:
            raise NotImplementedError
        
        assert (pre_norm or post_norm)
        self.prenorm = PreNorm if pre_norm else lambda dim, fn, context_dim=None: fn
        self.postnorm = PostNorm if post_norm else nn.Identity
        
        ff = ThinFeedForward if thin_ff else FeedForward
        
        # * decoder cross attention layers
        get_cross_attn = \
            lambda: self.prenorm(
                latent_dim, 
                Attention(
                    latent_dim, input_dim,
                    heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout, attn_type = cross_attn_type, more_dropout = more_dropout, xavier_init = xavier_init
                ), 
                context_dim = input_dim)
        get_cross_ff = lambda: self.prenorm(latent_dim, ff(latent_dim, dropout = ff_dropout, activation = activation, mult=ff_mult, more_dropout = more_dropout, xavier_init = xavier_init))
        get_cross_postnorm = lambda: self.postnorm(latent_dim)
        
        # * self attention of queries (first self attention layer of decoder)
        get_latent_attn = lambda: self.prenorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout, more_dropout = more_dropout, xavier_init = xavier_init))
        get_latent_ff = lambda: self.prenorm(latent_dim, ff(latent_dim, dropout = ff_dropout, activation = activation, mult=ff_mult, more_dropout = more_dropout, xavier_init = xavier_init))
        get_latent_postnorm = lambda: self.postnorm(latent_dim)
        
        # * encoder layers
        # FIXME add option to encoder layers to have its own hyper-parameter option, not just following latent layer options
        get_pre_self_attn = lambda: self.prenorm(input_dim, Attention(input_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout, more_dropout = more_dropout, xavier_init = xavier_init))
        get_pre_self_ff = lambda: self.prenorm(input_dim, ff(input_dim, dropout = ff_dropout, activation = activation, mult=ff_mult, more_dropout = more_dropout, xavier_init = xavier_init))
        get_pre_self_postnorm = lambda: self.postnorm(input_dim)

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff, get_pre_self_attn, get_pre_self_ff = map(cache_fn, \
            (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff, get_pre_self_attn, get_pre_self_ff)
        )

        self.layers = nn.ModuleList([])
        
        # self attention before going into decoder, coresponding to the DETR encoder
        self.pre_self_attns = nn.ModuleList([])
        for _ in range(self_before_cross_attn):
            self.pre_self_attns.append(nn.ModuleList([
                get_pre_self_attn(**{'_cache': False}),
                get_pre_self_postnorm(),
                get_pre_self_ff(**{'_cache': False}),
                get_pre_self_postnorm()
            ]))
        
        # self attention for decoder query (not necessary but following DETR's choice)
        self.query_self_attns = nn.ModuleList([])
        for _ in range(query_self_attn):
            self.query_self_attns.append(nn.ModuleList([
                get_latent_attn(**{'_cache': False}),
                get_latent_postnorm(),
                get_latent_ff(**{'_cache': False}),
                get_latent_postnorm()
            ]))

        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}
            
            # self attention after cross attention, only exists in perceiver arch.
            self_attns = nn.ModuleList([])
            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_postnorm(),
                    get_latent_ff(**cache_args),
                    get_latent_postnorm()
                ]))
            
            # cross attention layer, DETR decoder
            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_postnorm(),
                get_cross_ff(**cache_args),
                get_cross_postnorm(),
                self_attns
            ]))

        # Last FC layer
        if not last_fc:
            assert latent_dim == self.num_classes
        self.last_layer = nn.Sequential(
            nn.LayerNorm(latent_dim) if last_ln and not post_norm else nn.Identity(),
            nn.Linear(latent_dim, num_classes) if last_fc else nn.Identity()
        )
        
        self.encoder_output_holder = nn.Identity()
        self.decoder_output_holder = nn.Identity()
        
    def get_queries(self, b):
        if self.query_type == 'learned':
            ret = repeat(self.latents, 'n d -> b n d', b = b)
        elif self.query_type == 'slot':
            slots_init = torch.randn((b, self.num_latents, self.latent_dim)).cuda()
            ret = self.slots_mu + self.slots_log_sigma.exp() * slots_init
        return ret

    def forward(self, data, res, mask = None):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # concat to channels of data and flatten axis
        pos = self.pos_enc(data)
        
        data = rearrange(data, 'b ... d -> b (...) d')
        
        x = self.get_queries(b).type_as(data)
        
        # layers
        for pre_self_attn, pn1, self_ff, pn2 in self.pre_self_attns:
            data = pre_self_attn(data, mask = mask, q_pos = pos, k_pos = pos) + data
            data = pn1(data)
            data = self_ff(data) + data
            data = pn2(data)
        
        # concat global feature to the encoder output -> which acts as a HINT for the decoder
        data = torch.cat([data, rearrange(res, 'b d -> b 1 d')], dim=1)
        mask = torch.cat([mask, torch.zeros((b, 1)).type_as(mask)], dim=1)
        pos = pos if pos is None else torch.cat([pos, pos.mean(dim=1, keepdim=True)], dim=1)
        
        data = self.encoder_output_holder(data)
        
        for query_self_attn, pn1, self_ff, pn2 in self.query_self_attns:
            x = query_self_attn(x) + x
            x = pn1(x)
            x = self_ff(x) + x
            x = pn2(x)

        for cross_attn, pn1, cross_ff, pn2, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask, k_pos = pos, q_pos = None) + x
            x = pn1(x)
            x = cross_ff(x) + x
            x = pn2(x)
            
            # only for perceiver arch. not used in current implementation
            for self_attn, pn1, self_ff, pn2 in self_attns:
                x = self_attn(x) + x
                x = pn1(x)
                x = self_ff(x) + x
                x = pn2(x)
                
        x = self.decoder_output_holder(x)
        
        # last_attn_map = rearrange(last_attn_map, '(b h) n d -> b h n d', h = 8).mean(dim=1)
        return self.last_layer(x)
