import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence 
from torch.distributions import Normal
from distance import SetwiseDistance
from einops import rearrange, repeat

def cosine_sim(x, y):
    """Cosine similarity between all the image and sentence pairs. Assumes x and y are l2 normalized"""
    return x.mm(y.t())

def order_sim(x, y):
    """Order embeddings similarity measure $max(0, x-y)$"""
    YmX = (y.unsqueeze(1).expand(y.size(0), x.size(0), y.size(1)) - \
                    x.unsqueeze(0).expand(y.size(0), x.size(0), y.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

# Problematic: could induce NaN loss
def l2norm_old(x):
    """L2-normalize columns of x"""
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    return torch.div(x, norm)

def l2norm(x):
    """L2-normalize columns of x"""
    return F.normalize(x, p=2, dim=-1)

def rbf(x, y, gamma):
    """RBF kernel K(x,y) """
    pdist = torch.norm(x[:, None] - y, dim=2, p=2)
    return torch.exp(-gamma * pdist)

def rbf_memory_efficient(x, y, gamma):
    """RBF kernel that does not cause memory shortage"""
    cdist = torch.cdist(x, y)
    return torch.exp(-gamma * cdist)

class PVSELoss(nn.Module):

    def __init__(self, opt, reduction='mean'):
        super(PVSELoss, self).__init__()

        self.margin = opt.margin if hasattr(opt, 'margin') else 1.0
        self.num_embeds = opt.img_num_embeds if hasattr(opt, 'img_num_embeds') else 1
        self.mmd_weight = opt.mmd_weight if hasattr(opt, 'mmd_weight') else 0.
        self.div_weight = opt.div_weight if hasattr(opt, 'div_weight') else 0.
        self.sim_fn = order_sim if hasattr(opt, 'order') and opt.order else cosine_sim
        self.max_violation = opt.max_violation if hasattr(opt, 'max_violation') else False
        self.reduction = reduction

        if self.num_embeds > 1:
            self.max_pool = torch.nn.MaxPool2d(self.num_embeds)


    def diversity_loss(self, x):
        x = l2norm(x) # Columns of x MUST be l2-normalized
        gram_x = x.bmm(x.transpose(1,2))
        I = torch.autograd.Variable((torch.eye(x.size(1)) > 0.5).repeat(gram_x.size(0), 1, 1))
        if torch.cuda.is_available():
            I = I.cuda()
        gram_x.masked_fill_(I, 0.0)
        loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (self.num_embeds**2)
        return loss.mean() if self.reduction=='mean' else loss.sum()


    def mmd_rbf_loss(self, x, y, gamma=None):
        if gamma is None:
            gamma = 1./x.size(-1)
        loss = rbf(x, x, gamma) - 2 * rbf(x, y, gamma) + rbf(y, y, gamma)
        return loss.mean() if self.reduction=='mean' else loss.sum()


    def triplet_ranking_loss(self, A, B, I, max_dim):
        loss = (self.margin + A - B).clamp(min=0.0)
        loss.masked_fill_(I, 0.0)
        if self.max_violation:
            loss = loss.max(max_dim)[0]
        return loss.mean() if self.reduction=='mean' else loss.sum()


    def forward(self, img, txt, img_r, txt_r):
        loss, losses = 0, dict()

        # compute image-sentence score matrix
        if self.num_embeds > 1:
            scores = self.sim_fn(img.view(-1, img.size(-1)), txt.view(-1, txt.size(-1)))
            scores = self.max_pool(scores.unsqueeze(0)).squeeze()
        else:
            scores = self.sim_fn(img, txt)

        diagonal = scores.diag().view(img.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        mask = torch.eye(scores.size(0)) > .5
        I = torch.autograd.Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()

        # compare every diagonal score to scores in its column (image-to-text retrieval)
        i2t_loss = self.triplet_ranking_loss(scores, d1, I, 1)

        # compare every diagonal score to scores in its row (text-to-image retrieval)
        t2i_loss = self.triplet_ranking_loss(scores, d2, I, 0)

        ranking_loss = i2t_loss + t2i_loss
        loss += ranking_loss
        losses['ranking_loss'] = ranking_loss

        # diversity loss
        if self.num_embeds > 1 and self.div_weight > 0.:
            div_loss = self.diversity_loss(img_r) + self.diversity_loss(txt_r)
            loss += self.div_weight * div_loss
            losses['div_loss'] = div_loss

        # domain discrepancy loss
        if self.num_embeds > 1 and self.mmd_weight > 0.:
            mmd_loss = self.mmd_rbf_loss(img.view(-1, img.size(-1)), txt.view(-1, txt.size(-1)), gamma=0.5)
            loss += self.mmd_weight * mmd_loss
            losses['mmd_loss'] = mmd_loss

        return loss, losses
    
class PCMELoss(nn.Module):

    def __init__(self, opt, reduction='mean'):
        super(PCMELoss, self).__init__()

        self.margin = opt.margin if hasattr(opt, 'margin') else 1.0
        self.num_embeds = opt.num_embeds if hasattr(opt, 'num_embeds') else 1
        self.reduction = reduction
        self.kl_weight = opt.kl_weight if hasattr(opt, 'kl_weight') else 0.00001
        self.unif_weight = opt.unif_weight if hasattr(opt, 'unif_weight') else 0.
        self.img_emb_norm = []
        self.txt_emb_norm = []
        
        if self.num_embeds > 1:
            self.avg_pool = torch.nn.AvgPool2d(self.num_embeds)


    def kl_loss(self, x: Normal):
        # import ipdb; ipdb.set_trace()
        prior = Normal(torch.zeros_like(x.mean), torch.ones_like(x.variance))
        loss = kl_divergence(x, prior).sum(dim=-1)
        return loss.mean()
    
    
    def unif_loss(self, img_embs, txt_embs):
        # import ipdb; ipdb.set_trace()
        total_embs = torch.cat([img_embs, txt_embs], dim=0)
        exp_dist = torch.exp(torch.cdist(total_embs, total_embs, p=2).pow(2) * (-2))
        return torch.sum(exp_dist) - torch.trace(exp_dist)
    
    
    def soft_contrastive_loss(self, img_embs, txt_embs, a, b):
        dist = torch.cdist(img_embs, txt_embs)
        match_prob = torch.sigmoid((-a) * dist + b)
        factorized_match_prob = self.avg_pool(match_prob.unsqueeze(0)).squeeze(0)
        mask = torch.eye(factorized_match_prob.shape[0]).cuda()
        pos_term = torch.where(mask==1, -torch.log(factorized_match_prob), torch.zeros_like(factorized_match_prob))
        neg_term = torch.where(mask==0, -torch.log(1 - factorized_match_prob), torch.zeros_like(factorized_match_prob))
        return (pos_term + neg_term).mean()


    def forward(self, img_embs, txt_embs, img_mean, txt_mean, img_var, txt_var, a, b):
        loss, losses = 0, dict()
        # import ipdb; ipdb.set_trace()

        # compare every diagonal score to scores in its column (image-to-text retrieval)
        self.img_emb_norm += [img_embs.reshape(-1, img_embs.shape[-1]).norm(dim=1).mean().item()]
        self.txt_emb_norm += [txt_embs.reshape(-1, txt_embs.shape[-1]).norm(dim=1).mean().item()]
        
        loss += self.soft_contrastive_loss(img_embs.reshape(-1, img_embs.shape[-1]), txt_embs.reshape(-1, txt_embs.shape[-1]), a, b)

        losses['soft_contrastive_loss'] = loss

        if self.num_embeds > 1 and self.kl_weight > 0.:
            kl_loss = self.kl_loss(Normal(torch.cat([img_mean, txt_mean], dim=0), torch.cat([img_var, txt_var], dim=0)))
            loss += self.kl_weight * kl_loss
            losses['kl_loss'] = kl_loss

        if self.num_embeds > 1 and self.unif_weight > 0.:
            unif_loss = self.unif_loss(img_embs.reshape(-1, img_embs.shape[-1]), txt_embs.reshape(-1, txt_embs.shape[-1]))
            loss += self.unif_weight * unif_loss
            losses['unif_loss'] = unif_loss

        return loss, losses

class ChamferLoss(nn.Module):

    def __init__(self, opt, reduction='mean', smooth=False):
        super(ChamferLoss, self).__init__()

        self.margin = opt.margin if hasattr(opt, 'margin') else 1.0
        self.num_embeds = opt.num_embeds if hasattr(opt, 'num_embeds') else 1
        self.reduction = reduction
        self.img_emb_norm = []
        self.txt_emb_norm = []
        self.margin = opt.margin if hasattr(opt, 'margin') else 1.0
        self.mmd_weight = opt.mmd_weight if hasattr(opt, 'mmd_weight') else 0.
        self.div_weight = opt.div_weight if hasattr(opt, 'div_weight') else 0.
        self.max_violation = opt.max_violation if hasattr(opt, 'max_violation') else False
        self.smooth = smooth
        self.i2t_weight, self.t2i_weight = opt.i2t_weight, opt.t2i_weight
        
        self.x_axis_max_pool = torch.nn.MaxPool2d((1, self.num_embeds))
        self.x_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(1, self.num_embeds))
        self.y_axis_max_pool = torch.nn.MaxPool2d((self.num_embeds, 1))
        self.y_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(self.num_embeds, 1))
            
    def diversity_loss(self, x):
        x = l2norm(x) # Columns of x MUST be l2-normalized
        gram_x = x.bmm(x.transpose(1,2))
        I = torch.autograd.Variable((torch.eye(x.size(1)) > 0.5).repeat(gram_x.size(0), 1, 1))
        if torch.cuda.is_available():
            I = I.cuda()
        gram_x.masked_fill_(I, 0.0)
        loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (self.num_embeds**2)
        return loss.mean() if self.reduction=='mean' else loss.sum()


    def mmd_rbf_loss(self, x, y, gamma=None):
        if gamma is None:
            gamma = 1./x.size(-1)
        loss = rbf(x, x, gamma) - 2 * rbf(x, y, gamma) + rbf(y, y, gamma)
        return loss.mean() if self.reduction=='mean' else loss.sum()
    
    def smoothed_chamfer_distance_euclidean(self, img_embs, txt_embs):
        """
            Method to compute Chamfer distance between image embeddings and text embeddings.
            Left term is for i -> t, and right term is for t -> i.
        """
        dist = torch.cdist(img_embs, txt_embs)
        
        right_term = -self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(-dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = -self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(-dist.unsqueeze(0))
            ))).squeeze(0)
        
        smoothed_chamfer_dist = (right_term + left_term) / (2 * self.num_embeds)

        return smoothed_chamfer_dist
    
    def smoothed_chamfer_distance_cosine(self, img_embs, txt_embs):
        """
            Method to compute Chamfer distance between image embeddings and text embeddings.
            Left term is for i -> t, and right term is for t -> i.
        """
        dist = cosine_sim(img_embs, txt_embs)
        
        right_term = self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(dist.unsqueeze(0))
            ))).squeeze(0)
        
        smoothed_chamfer_dist = (right_term + left_term) / (2 * self.num_embeds)

        return smoothed_chamfer_dist
    
    def chamfer_distance(self, img_embs, txt_embs):
        """
            Method to compute Smoothed Chafer Distance(SCD).
            Max pool is changed to LSE.
        """
        # dist = torch.cdist(img_embs, txt_embs)
        
        # right_term = -self.y_axis_sum_pool(self.x_axis_max_pool(-dist.unsqueeze(0))).squeeze(0)
        # left_term = -self.x_axis_sum_pool(self.y_axis_max_pool(-dist.unsqueeze(0))).squeeze(0)
        
        dist = cosine_sim(img_embs, txt_embs)
        
        right_term = self.y_axis_sum_pool(self.x_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        left_term = self.x_axis_sum_pool(self.y_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        
        chamfer_dist = (right_term + left_term) / (2 * self.num_embeds)

        return chamfer_dist
    
    def triplet_ranking_loss(self, A, B, I, max_dim):
        loss = (self.margin + A - B).clamp(min=0.0)
        loss.masked_fill_(I, 0.0)
        if self.max_violation:
            loss = loss.max(max_dim)[0]
        return loss.mean()
    

    def forward(self, img_embs, txt_embs, img_r, txt_r):
        loss, losses = 0, dict()

        # compare every diagonal score to scores in its column (image-to-text retrieval)
        self.img_emb_norm += [img_embs.reshape(-1, img_embs.shape[-1]).norm(dim=1).mean().item()]
        self.txt_emb_norm += [txt_embs.reshape(-1, txt_embs.shape[-1]).norm(dim=1).mean().item()]
        img_embs = img_embs.reshape(-1, img_embs.shape[-1])
        txt_embs = txt_embs.reshape(-1, txt_embs.shape[-1])
        if self.smooth:
            setwise_dist = self.smoothed_chamfer_distance_cosine(img_embs, txt_embs)
        else:
            setwise_dist = self.chamfer_distance(img_embs, txt_embs)
        diagonal = setwise_dist.diag().reshape(-1, 1)
        i2t_pos = diagonal.expand_as(setwise_dist)
        t2i_pos = diagonal.t().expand_as(setwise_dist)

        mask = (torch.eye(setwise_dist.shape[0]) > .5).cuda()

        i2t_loss = self.triplet_ranking_loss(setwise_dist, i2t_pos, mask, 1)
        t2i_loss = self.triplet_ranking_loss(setwise_dist, t2i_pos, mask, 0)
        
        losses['i2t_loss'] = i2t_loss
        losses['t2i_loss'] = t2i_loss
        loss += self.i2t_weight * i2t_loss + self.t2i_weight * t2i_loss
        
        
        # setwise_dist.register_hook(lambda x : print("chamfer loss backward", x.nonzero()))
        debug_dict = {}
        debug_dict['i2t_pos'] = i2t_pos
        debug_dict['t2i_pos'] = t2i_pos
        debug_dict['setwise_dist'] = setwise_dist
        debug_dict['mask'] = mask
        debug_dict['diagonal'] = diagonal
        # setwise_dist.register_hook(lambda x : debug_dict.update({'grad':x}))
        
        
        if self.num_embeds > 1 and self.div_weight > 0.:
            div_loss = self.diversity_loss(img_r) + self.diversity_loss(txt_r)
            loss += self.div_weight * div_loss
            losses['div_loss'] = div_loss

        # domain discrepancy loss
        if self.num_embeds > 1 and self.mmd_weight > 0.:
            mmd_loss = self.mmd_rbf_loss(img_embs, txt_embs, gamma=0.5)
            loss += self.mmd_weight * mmd_loss
            losses['mmd_loss'] = mmd_loss

        return loss, losses, debug_dict

class AsymmetricTripletLoss(nn.Module):
    def __init__(self, img_set_size, txt_set_size, distance_fn, opt, reduction='mean', txt_per_img=5, is_cosine=True):
        super(AsymmetricTripletLoss, self).__init__()
        
        # loss hyperparameters
        self.margin = opt.margin if hasattr(opt, 'margin') else 1.0
        self.img_num_embeds = opt.img_num_embeds
        self.txt_num_embeds = opt.txt_num_embeds
        self.reduction = reduction
        self.img_emb_norm = []
        self.txt_emb_norm = []
        self.margin = opt.margin if hasattr(opt, 'margin') else 1.0
        self.mmd_weight = opt.mmd_weight if hasattr(opt, 'mmd_weight') else 0.
        self.div_weight = opt.div_weight if hasattr(opt, 'div_weight') else 0.
        self.unif_weight = opt.unif_weight if hasattr(opt, 'unif_weight') else 0.
        self.qreg_weight = opt.qreg_weight if hasattr(opt, 'qreg_weight') else 0.
        self.max_violation = opt.max_violation if hasattr(opt, 'max_violation') else False
        self.unif_residual = opt.unif_residual
        
        # set_distance hyperparameters
        self.distance_fn = distance_fn
        self.is_cosine = is_cosine
        self.img_set_size, self.txt_set_size = img_set_size, txt_set_size
        self.txt_per_img = txt_per_img
        """
        set_per_img : Matching sets per each image. 2 scenarios.
        Lets assumes that 
            K : number of embeddings for each sample, 
            T : number of matching captions per image (5 in COCO).
        set_per_img = (T * K) / txt_set_size
        1. set_per_img = T, txt_set_size = K
        2. set_per_img = 1, txt_set_size = T * K
        """
        self.set_per_img = int(self.txt_per_img * self.txt_num_embeds / self.txt_set_size) 
        self.i2t_weight, self.t2i_weight = opt.i2t_weight, opt.t2i_weight
        self.semi_hard_triplet = opt.semi_hard_triplet
        
    def diversity_loss(self, x, num_embeds):
        if num_embeds == 1:
            return 0.0
        x = l2norm(x) # Columns of x MUST be l2-normalized
        gram_x = x.bmm(x.transpose(1,2))
        I = torch.autograd.Variable((torch.eye(x.size(1)) > 0.5).repeat(gram_x.size(0), 1, 1))
        if torch.cuda.is_available():
            I = I.cuda()
        gram_x.masked_fill_(I, 0.0)
        loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (num_embeds**2)
        return loss.mean() if self.reduction=='mean' else loss.sum()

    def mmd_rbf_loss(self, x, y, gamma=None):
        if gamma is None:
            gamma = 1./x.size(-1)
        if self.reduction=='mean':
            loss = rbf_memory_efficient(x, x, gamma).mean() - 2 * rbf_memory_efficient(x, y, gamma).mean() + rbf_memory_efficient(y, y, gamma).mean()
        else:
            loss = rbf_memory_efficient(x, x, gamma).sum() - 2 * rbf_memory_efficient(x, y, gamma).sum() + rbf_memory_efficient(y, y, gamma).sum()
        return loss
    
    def batchwise_uniformity_loss(self, embs, num_embeds, t=2):
        if num_embeds == 1:
            return 0.0
        rbf = torch.exp(-t * torch.cdist(embs, embs).pow(2))
        I = torch.autograd.Variable(repeat(
            torch.triu(torch.ones(rbf.shape[1], rbf.shape[1]), diagonal=1), 
            'n d -> b n d', 
            b=rbf.shape[0]
        )).cuda()
        rbf = torch.where(I == 1, rbf, torch.zeros_like(rbf))
        loss = torch.stack([r.sum() for r in rbf]) / (num_embeds * (num_embeds - 1) * 0.5)
        return loss.mean()
    
    def query_regularizer(self, img_query, txt_query, t=2):
        return torch.exp(-t * F.pdist(img_query)).mean() + torch.exp(-t * F.pdist(txt_query)).mean()
    
    def triplet_ranking_loss(self, A, B, max_dim):
        if self.semi_hard_triplet:
            loss = (self.margin + A - B).clamp(min=0.0, max=self.margin)
            num_triplets = torch.nonzero(loss).shape[0]
            if num_triplets == 0:
                return loss.mean()
            else:
                return loss.sum() / num_triplets
        else:
            loss = (self.margin + A - B).clamp(min=0.0)
            if self.max_violation:
                loss = loss.max(max_dim)[0]
            return loss.mean()
    
    def forward(self, img_embs, txt_embs, img_r, txt_r, img_query=None, txt_query=None):
        loss, losses = 0, dict()
        
        # compare every diagonal score to scores in its column (image-to-text retrieval)
        self.img_emb_norm += [img_embs.reshape(-1, img_embs.shape[-1]).norm(dim=1).mean().item()]
        self.txt_emb_norm += [txt_embs.reshape(-1, txt_embs.shape[-1]).norm(dim=1).mean().item()]
        
        # reshape embeddings as 2D tensors (given as 3D tensors).
        img_embs = img_embs.reshape(-1, img_embs.shape[-1])
        txt_embs = txt_embs.reshape(-1, txt_embs.shape[-1])
        
        # Compute setwise distance with provided set distance metric
        setwise_dist = self.distance_fn(self.is_cosine, img_embs, txt_embs)
        
        # generate mask based on the computed number of sets per images
        mask = (torch.eye(setwise_dist.shape[0]) > .5).cuda()
        mask = mask.view(-1, 1).repeat(1, self.set_per_img).reshape(setwise_dist.shape)
        
        """
        example when txt_set_size : 5, set_per_img : 5>
            coco dataset, caption per image : 5 
            setwise_dist : (128, 640)
            mask : (128, 640) , indicates matching pairs
            i2t_pos : (5, 128, 1) 
            i2t_neg : (5, 128, 635) , repeat of (1, 128, 635)
        """
        
        neg_mask = ~mask
        # i2t loss. multiple matching captions exist for each each image
        i2t_pos = setwise_dist[mask].view(setwise_dist.shape[0], -1, 1).permute(1, 0, 2)
        i2t_neg = setwise_dist[neg_mask].view(1, setwise_dist.shape[0], -1)
        i2t_loss = self.triplet_ranking_loss(i2t_neg, i2t_pos, 2)
        
        # t2i loss. single matching image exists for each each caption
        t2i_pos = setwise_dist.t()[mask.t()].reshape(setwise_dist.shape[1], -1)
        t2i_neg = setwise_dist.t()[neg_mask.t()].reshape(setwise_dist.shape[1], -1)
        t2i_loss = self.triplet_ranking_loss(t2i_neg, t2i_pos, 1)
        
        losses['t2i_loss'] = t2i_loss
        losses['i2t_loss'] = i2t_loss 
        loss += self.i2t_weight * i2t_loss + self.t2i_weight * t2i_loss
        
        if self.div_weight > 0.:
            div_loss = self.diversity_loss(img_r, self.img_num_embeds) + \
                self.diversity_loss(txt_r, self.txt_num_embeds)
            loss += self.div_weight * div_loss
            losses['div_loss'] = div_loss
        
        # domain discrepancy loss
        if self.mmd_weight > 0.:
            mmd_loss = self.mmd_rbf_loss(img_embs, txt_embs, gamma=0.5)
            loss += self.mmd_weight * mmd_loss
            losses['mmd_loss'] = mmd_loss
            
        if self.unif_weight > 0.:
            unif_img = l2norm(img_r) if self.unif_residual else img_embs
            unif_txt = l2norm(txt_r) if self.unif_residual else txt_embs
            unif_loss = self.batchwise_uniformity_loss(unif_img.reshape(-1, self.img_num_embeds, unif_img.shape[-1]), self.img_num_embeds) + \
                self.batchwise_uniformity_loss(unif_txt.reshape(-1, self.txt_num_embeds, unif_txt.shape[-1]), self.txt_num_embeds)
            loss += self.unif_weight * unif_loss
            losses['unif_loss'] = unif_loss
            
        if self.qreg_weight > 0. and (img_query is not None):
            qreg = self.query_regularizer(img_query, txt_query)
            loss += self.qreg_weight * qreg
            losses['query_regularizer'] = qreg

        return loss, losses