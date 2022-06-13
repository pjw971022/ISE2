import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

def cosine_sim(x, y):
    """Cosine similarity between all the image and sentence pairs. Assumes that x and y are l2 normalized"""
    return x.mm(y.t())

class MPdistance(nn.Module):
    def __init__(self, avg_pool):
        super(MPdistance, self).__init__()
        self.avg_pool = avg_pool
        self.alpha, self.beta = nn.Parameter(torch.ones(1)).cuda(), nn.Parameter(torch.zeros(1)).cuda()
        
    def forward(self, img_embs, txt_embs):
        dist = torch.cdist(img_embs, txt_embs)
        avg_distance = self.avg_pool(torch.sigmoid(self.alpha * dist.unsqueeze(0) + self.beta)).squeeze(0)
        return avg_distance
    
class SetwiseDistance(nn.Module):
    def __init__(self, img_set_size, txt_set_size, denominator, temperature=1, temperature_txt_scale=1, alpha=0.1):
        super(SetwiseDistance, self).__init__()
        # poolings
        self.img_set_size = img_set_size
        self.txt_set_size = txt_set_size
        self.denominator = denominator
        self.temperature = temperature
        self.temperature_txt_scale = temperature_txt_scale # used when computing i2t distance
        self.alpha = alpha
        self.xy_max_pool = torch.nn.MaxPool2d((self.img_set_size, self.txt_set_size))
        self.xy_avg_pool = torch.nn.AvgPool2d((self.img_set_size, self.txt_set_size))
        self.x_axis_max_pool = torch.nn.MaxPool2d((1, self.txt_set_size))
        self.x_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(1, self.txt_set_size))
        self.y_axis_max_pool = torch.nn.MaxPool2d((self.img_set_size, 1))
        self.y_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(self.img_set_size, 1))
        
        self.x_axis_sum_pool2 = torch.nn.LPPool2d(norm_type=1, kernel_size=(1, self.img_set_size))
        self.y_axis_sum_pool2 = torch.nn.LPPool2d(norm_type=1, kernel_size=(self.txt_set_size, 1))
        
        self.x_axis_max_pool2 = torch.nn.MaxPool2d((1, self.img_set_size))
        self.y_axis_max_pool2 = torch.nn.MaxPool2d((self.txt_set_size, 1))
        
        self.mp_dist = MPdistance(self.xy_avg_pool)
        
    def smooth_chamfer_distance_euclidean(self, img_embs, txt_embs):
        """
            Method to compute Smooth Chafer Distance(SCD). Max pool is changed to LSE.
            Use euclidean distance(L2-distance) to measure distance between elements.
        """
        dist = torch.cdist(img_embs, txt_embs)
        
        right_term = -self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(-self.temperature * self.temperature_txt_scale * dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = -self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(-self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = (right_term / (self.img_set_size * self.temperature * self.temperature_txt_scale) + left_term / (self.txt_set_size * self.temperature)) / (self.denominator)

        return smooth_chamfer_dist
    
        
    def smooth_chamfer_distance_cosine_self(self, img_embs, txt_embs):
        """
            cosine version of smooth_chamfer_distance_euclidean(img_embs, txt_embs)
        """
        dist = cosine_sim(img_embs, txt_embs)
        dist_iself = cosine_sim(img_embs, img_embs)
        dist_tself = cosine_sim(txt_embs, txt_embs)
        dist_iself = dist_iself - torch.diag(dist_iself)
        dist_tself = dist_tself - torch.diag(dist_tself)
        iself_term = self.alpha*self.x_axis_sum_pool(self.y_axis_sum_pool2(dist_iself.unsqueeze(0))).squeeze(0) 
        tself_term = self.alpha*self.x_axis_sum_pool2(self.y_axis_sum_pool(dist_tself.unsqueeze(0))).squeeze(0)
        
        iself_term = torch.diag(iself_term, 0).unsqueeze(0).t().expand(int(img_embs.shape[0]/self.img_set_size), int(txt_embs.shape[0]/self.txt_set_size))
        tself_term = torch.diag(tself_term, 0).unsqueeze(0).expand(int(img_embs.shape[0]/self.img_set_size), int(txt_embs.shape[0]/self.txt_set_size))
        

        right_term = self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(self.temperature * self.temperature_txt_scale * dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = (right_term / (self.img_set_size * self.temperature * self.temperature_txt_scale) + left_term / (self.txt_set_size * self.temperature)) / (self.denominator)
        self_term  = self.alpha*(iself_term + tself_term)

        return smooth_chamfer_dist + self_term

    def smooth_chamfer_distance_cosine(self, img_embs, txt_embs):
        """
            cosine version of smooth_chamfer_distance_euclidean(img_embs, txt_embs)
        """
        dist = cosine_sim(img_embs, txt_embs)
            
        right_term = self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(self.temperature * self.temperature_txt_scale * dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = (right_term / (self.img_set_size * self.temperature * self.temperature_txt_scale) + left_term / (self.txt_set_size * self.temperature)) / (self.denominator)

        return smooth_chamfer_dist
    
    def smooth_chamfer_distance_euclidean_t2i(self, img_embs, txt_embs):
        """
            Method to compute Smooth Chafer Distance(SCD). Max pool is changed to LSE.
            Use euclidean distance(L2-distance) to measure distance between elements.
        """
        dist = torch.cdist(img_embs, txt_embs)
        
        left_term = -self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(-self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = left_term / (self.txt_set_size * self.temperature * self.denominator)

        return smooth_chamfer_dist
    
    def smooth_chamfer_distance_cosine_t2i(self, img_embs, txt_embs):
        """
            cosine version of smooth_chamfer_distance_euclidean(img_embs, txt_embs)
        """
        dist = cosine_sim(img_embs, txt_embs)
        
        left_term = self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = left_term / (self.txt_set_size * self.temperature * self.denominator)

        return smooth_chamfer_dist
    
    def chamfer_distance_euclidean(self, img_embs, txt_embs):
        """
            Method to compute Chamfer distance between image embeddings and text embeddings.
            Left term is for i -> t, and right term is for t -> i.
        """
        dist = torch.cdist(img_embs, txt_embs)
        
        right_term = -self.y_axis_sum_pool(self.x_axis_max_pool(-dist.unsqueeze(0))).squeeze(0)
        left_term = -self.x_axis_sum_pool(self.y_axis_max_pool(-dist.unsqueeze(0))).squeeze(0)
        
        chamfer_dist = (right_term / self.img_set_size + left_term / self.txt_set_size) / self.denominator

        return chamfer_dist
    
    def chamfer_distance_cosine(self, img_embs, txt_embs):
        """
            cosine version of chamfer_distance_euclidean(img_embs, txt_embs)
        """
        dist = cosine_sim(img_embs, txt_embs)
        
        right_term = self.y_axis_sum_pool(self.x_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        left_term = self.x_axis_sum_pool(self.y_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        
        chamfer_dist = (right_term / self.img_set_size + left_term / self.txt_set_size) / self.denominator

        return chamfer_dist
    
    def max_distance_euclidean(self, img_embs, txt_embs):
        dist = torch.cdist(img_embs, txt_embs)
        max_distance = -self.xy_max_pool(-dist.unsqueeze(0)).squeeze(0)
        return max_distance
    
    def max_distance_cosine(self, img_embs, txt_embs):
        dist = cosine_sim(img_embs, txt_embs)
        max_distance = self.xy_max_pool(dist.unsqueeze(0)).squeeze(0)
        return max_distance
    
    def only_first_cosine(self, is_cosine, img_embs, txt_embs):
        dist = cosine_sim(img_embs[0::2], txt_embs)
        # max_distance = self.x_axis_sum_pool(dist.unsqueeze(0)).squeeze(0)
        self.__init__(1, 2, 2, 16, 1)
        debug_term = self.smooth_chamfer_distance_cosine(img_embs[0::2], txt_embs)
        self.__init__(2, 2, 2, 16, 1)
        return debug_term
    
    def only_second_cosine(self, is_cosine, img_embs, txt_embs):
        dist = cosine_sim(img_embs[1::2], txt_embs)
        # max_distance = self.x_axis_sum_pool(dist.unsqueeze(0)).squeeze(0)
        self.__init__(1, 2, 2, 16, 1)
        debug_term = self.smooth_chamfer_distance_cosine(img_embs[1::2], txt_embs)
        self.__init__(2, 2, 2, 16, 1)
        return debug_term
    
    def only_first_txt_cosine(self, is_cosine, img_embs, txt_embs):
        dist = cosine_sim(img_embs, txt_embs[0::2])
        # max_distance = self.x_axis_sum_pool(dist.unsqueeze(0)).squeeze(0)
        self.__init__(2, 1, 2, 16, 1)
        debug_term = self.smooth_chamfer_distance_cosine(img_embs, txt_embs[0::2])
        self.__init__(2, 2, 2, 16, 1)
        return debug_term
    
    def only_second_txt_cosine(self, is_cosine, img_embs, txt_embs):
        dist = cosine_sim(img_embs, txt_embs[1::2])
        # max_distance = self.x_axis_sum_pool(dist.unsqueeze(0)).squeeze(0)
        self.__init__(2, 1, 2, 16, 1)
        debug_term = self.smooth_chamfer_distance_cosine(img_embs, txt_embs[1::2])
        self.__init__(2, 2, 2, 16, 1)
        return debug_term
    
    def smooth_chamfer_distance_dropped_cosine(self, img_embs, txt_embs, k):
        """ Drop except K elements in each set when computing distance between them
        """
        bi, d = img_embs.shape
        bt, d = txt_embs.shape
        
        dist = cosine_sim(img_embs, txt_embs)
        
        def k_max_pooling(x, k, dim):
            idx = x.topk(k, dim=dim)[1].sort(dim=dim)[0] 
            return x.gather(dim, idx).sum(dim=dim)
        
        lse_on_txts = torch.log(
            self.x_axis_sum_pool(
                torch.exp(self.temperature * self.temperature_txt_scale * dist.unsqueeze(0))
        )).squeeze(0)
        lse_on_imgs = torch.log(
            self.y_axis_sum_pool(
                torch.exp(self.temperature * dist.unsqueeze(0))
        )).squeeze(0)
                
        pooled_lse_on_txts = k_max_pooling(rearrange(lse_on_txts, '(bi n) bt -> bi n bt', n=self.img_set_size), k=k, dim=1)
        pooled_lse_on_imgs = k_max_pooling(rearrange(lse_on_imgs, 'bi (bt n) -> bi bt n', n=self.txt_set_size), k=k, dim=2)
        
        smooth_chamfer_dist_dropped = (pooled_lse_on_txts + pooled_lse_on_imgs) / (k * self.temperature * self.denominator)
        
        return smooth_chamfer_dist_dropped

    
    def smooth_chamfer_distance(self, is_cosine, img_embs, txt_embs):
        return self.smooth_chamfer_distance_cosine(img_embs, txt_embs) if is_cosine else self.smooth_chamfer_distance_euclidean(img_embs, txt_embs)
    
    def smooth_chamfer_self(self, is_cosine, img_embs, txt_embs):
        return self.smooth_chamfer_distance_cosine_self(img_embs, txt_embs) if is_cosine else self.smooth_chamfer_distance_euclidean(img_embs, txt_embs)
    
    def smooth_chamfer_distance_t2i(self, is_cosine, img_embs, txt_embs):
        return self.smooth_chamfer_distance_cosine_t2i(img_embs, txt_embs) if is_cosine else self.smooth_chamfer_distance_euclidean_t2i(img_embs, txt_embs)
    
    def chamfer_distance(self, is_cosine, img_embs, txt_embs):
        return self.chamfer_distance_cosine(img_embs, txt_embs) if is_cosine else self.chamfer_distance_euclidean(img_embs, txt_embs)
    
    def max_distance(self, is_cosine, img_embs, txt_embs):
        return self.max_distance_cosine(img_embs, txt_embs) if is_cosine else self.max_distance_euclidean(img_embs, txt_embs)
    
    def smooth_chamfer_distance_dropped(self, is_cosine, img_embs, txt_embs):
        return self.smooth_chamfer_distance_dropped_cosine(img_embs, txt_embs, 2)
    
    def avg_distance(self, is_cosine, img_embs, txt_embs):
        return self.mp_dist(img_embs, txt_embs)