import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform


#generator / policy 

def build_models(H): 
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            h = H.G_H
            self.core = nn.Sequential(nn.Linear(H.NOISE_DIM, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
            #define mu and sigma for numeric cols 
            self.mu = nn.Linear(h, len(H.NUM_COLS))
            self.log_sigma = nn.Parameter(torch.zeros(len(H.NUM_COLS)))  
            #categorical head
            self.cat_logits = nn.Linear(h, H.CAT_DIM)
            #value network 
            self.v = nn.Linear(h, 1)
        def sample(self, z):
            h = self.core(z)
            #NUMERIC 
            sigma = self.log_sigma.exp()
            num_dist = Normal(self.mu(h), sigma)
            if H.USE_TANH: 
                tanh_dist =  TransformedDistribution(num_dist, [TanhTransform(cache_size=1)])
                #rsample interally implements z = u + oe
                num_unit = tanh_dist.rsample() #(-1, 1)
                num = 0.5 * (num_unit + 1) #(0, 1) 
                num = num.clamp(H.EPS, 1.0 - H.EPS) #to avoid bugs 
                logp = tanh_dist.log_prob(num_unit).sum(-1) #e computed without explicitly storing it
            else: 
                #rsample interally implements z = u + oe
                num = num_dist.rsample() 
                logp = num_dist.log_prob(num).sum(-1)
            #CATEGORICAL 
            logits = self.cat_logits(h)
            cat_dist = Independent(Bernoulli(logits=logits), 1)
            cat = cat_dist.sample() 
            logp_cat = cat_dist.log_prob(cat) 
            logp += logp_cat
            row = torch.cat([num, cat], 1)
            #value head 
            val = self.v(h).squeeze()
            return row, logp, logp_cat, val

        def eval_action(self, z, row):
            h = self.core(z)
            #numeric 
            sigma = self.log_sigma.exp()
            if not H.USE_TANH: 
                num = row[:, :len(H.NUM_COLS)]
                num_log = Normal(self.mu(h), sigma).log_prob(num).sum(-1) 
            else:  
                num_scaled = row[:, :len(H.NUM_COLS)].clamp(H.EPS, 1.0-H.EPS) #avoids bugs 
                num_unit = 2 * num_scaled - 1 
                base_dist = Normal(self.mu(h), sigma) 
                tanh_dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])
                num_log = tanh_dist.log_prob(num_unit).sum(-1) 
            #categorical 
            logits = self.cat_logits(h)
            cat = row[:, len(H.NUM_COLS):]              
            cat_log = Independent(Bernoulli(logits=logits), 1).log_prob(cat)
            return num_log + cat_log, self.v(h).squeeze()

    #discriminator
    class Disc(nn.Module):
        def __init__(self):
            super().__init__()
            h = H.D_H
            self.fc = nn.Sequential(nn.Linear(len(H.NUM_COLS) + H.CAT_DIM, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, 1))
        def forward(self, row):
            return self.fc(row)
    
    return Policy().to(H.DEVICE), Disc().to(H.DEVICE)


