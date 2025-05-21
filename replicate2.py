import re
import random

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv

# ─────────────────────────────────────────────────────────────────────────────
# 1) SETUP & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    toks = text.split()
    toks = [w for w in toks if w not in stop_words]
    toks = [stemmer.stem(w) for w in toks]
    toks = [lemmatizer.lemmatize(w) for w in toks]
    return ' '.join(toks)

# Load data
api_df    = pd.read_excel('Web API.xls')
mashup_df = pd.read_excel('Mashup.xls')

# Rename to English
api_df = api_df.rename(columns={
    'API名称': 'api_name',
    'API标签': 'labels',
    'API服务描述': 'description'
})
mashup_df = mashup_df.rename(columns={
    '名称': 'mashup_name',
    '标签': 'labels',
    '描述': 'description',
    '相关API': 'used_apis',
    '类别': 'category'
})

# Assign IDs and node indices
a_pi = api_df.reset_index(drop=True)
a_pi['id'] = a_pi.index
m_sh = mashup_df.reset_index(drop=True)
m_sh['id'] = m_sh.index
num_mashups = len(m_sh)
num_apis    = len(a_pi)
m_sh['node_index'] = m_sh['id']
a_pi['node_index'] = a_pi['id'] + num_mashups

# Map API name → node_index
api_name_to_index = {row['api_name']: row['node_index']
                     for _, row in a_pi.iterrows()}

# Parse mashup→api associations
def parse_used(s):
    if pd.isna(s) or str(s).strip()=='' : return []
    names = [x.strip() for x in s.split(',') if x.strip()]
    return [api_name_to_index[n]
            for n in names if n in api_name_to_index]

mashup_used_apis = {row['node_index']: parse_used(row['used_apis'])
                    for _, row in m_sh.iterrows()}

# Combine & TF‑IDF
combined = pd.concat([m_sh, a_pi], ignore_index=True)
combined['desc_pp'] = combined['description'].fillna('').apply(preprocess_text)
vectorizer = TfidfVectorizer(max_features=128, min_df=2, max_df=0.9)
feats_np = vectorizer.fit_transform(combined['desc_pp']).toarray()
feats = torch.tensor(feats_np, dtype=torch.float)

# ─────────────────────────────────────────────────────────────────────────────
# 2) BUILD HETEROGENEOUS GRAPH
# ─────────────────────────────────────────────────────────────────────────────
data = HeteroData()
data['mashup'].x = feats[:num_mashups]
data['api'].x    = feats[num_mashups:]

# Build shortlist nodes
pairs = [(m,a) for m,alist in mashup_used_apis.items() for a in alist]
num_sl = len(pairs)
data['shortlist'].x = feats.new_zeros((num_sl, feats.size(1)))

# Helper to build edge_index
def build_edge_index(pairs, offset):
    src = [i for i,(m,a) in enumerate(pairs)]
    dst = [ (m if offset==0 else a-offset) for (m,a) in pairs ]
    return torch.tensor([src,dst], dtype=torch.long)

# mashup ←→ shortlist
data['shortlist','to_mashup','mashup'].edge_index = build_edge_index(pairs, 0)
data['mashup','rev_to_shortlist','shortlist'].edge_index = data['shortlist','to_mashup','mashup'].edge_index.flip(0)
# shortlist ←→ api
data['shortlist','to_api','api'].edge_index = build_edge_index(pairs, num_mashups)
data['api','rev_to_shortlist','shortlist'].edge_index   = data['shortlist','to_api','api'].edge_index.flip(0)

# ─────────────────────────────────────────────────────────────────────────────
# 3) LINK PREDICTION LOADER
# ─────────────────────────────────────────────────────────────────────────────
loader = LinkNeighborLoader(
    data,
    num_neighbors={et:[15,15] for et in data.edge_types},
    edge_label_index=(('shortlist','to_api','api'), data['shortlist','to_api','api'].edge_index),
    neg_sampling_ratio=1.0,
    batch_size=32,
    shuffle=True
)

# ─────────────────────────────────────────────────────────────────────────────
# 4) TIMBREGNN MODEL
# ─────────────────────────────────────────────────────────────────────────────
class TIMBREGNN(nn.Module):
    def __init__(self, data, hidden_channels=64, num_layers=3, feature_dims=None):
        super().__init__()
        self.node_types, self.edge_types = data.metadata()
        # Learnable embeddings per node type
        self.embeddings = nn.ModuleDict({
            n: nn.Embedding(data[n].num_nodes, hidden_channels)
            for n in self.node_types
        })
        # # Linear projections for nodes that have input features
        feature_dims = feature_dims or {}
        self.proj = nn.ModuleDict({
            n: nn.Linear(dim, hidden_channels)
            for n, dim in feature_dims.items()
        })
        # Stack relational SAGE layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({ et: SAGEConv((-1,-1), hidden_channels)
                                 for et in self.edge_types }, aggr='mean')
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, data):
        # Initialize each node’s embedding (learnable + optional feature)
        h = {}
        for ntype in self.node_types:
            n = data[ntype].num_nodes
            device = next(self.parameters()).device
            idx = torch.arange(n, device=device)
            emb = self.embeddings[ntype](idx)
            if ntype in self.proj and hasattr(data[ntype], 'x'):
                emb = emb + self.proj[ntype](data[ntype].x)
            h[ntype] = emb
        # message passing
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, data.edge_index_dict)
            for nt in h:
                h[nt] = norm(F.gelu(h[nt]))
        return h

    def score(self, h, edge_label_index, src_type='shortlist', dst_type='api'):
        src, dst = edge_label_index
        return F.cosine_similarity(h[src_type][src], h[dst_type][dst])

# ─────────────────────────────────────────────────────────────────────────────
# 5) TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TIMBREGNN(
    data, hidden_channels=64, num_layers=3,
    feature_dims={'mashup': feats.size(1), 'api': feats.size(1), 'shortlist': feats.size(1)}
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

EPOCHS = 40
for ep in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        h_dict = model(batch)
        # positive+neg edges
        eidx = batch['shortlist','to_api','api'].edge_label_index
        labels = batch['shortlist','to_api','api'].edge_label.float()
        scores = model.score(h_dict, eidx, 'shortlist', 'api')
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {ep:02d}/{EPOCHS} Loss: {total_loss/len(loader):.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    full_h = model(data.to(device))
    sl_h = full_h['shortlist'].cpu()
    api_h = full_h['api'].cpu()

HRs, NDCGs = [], []
for sid, (m,a) in enumerate(pairs):
    scores = F.cosine_similarity(sl_h[sid].unsqueeze(0), api_h, dim=1)
    top5 = torch.topk(scores, 5).indices.tolist()
    gt = a - num_mashups
    HRs.append(1.0 if gt in top5 else 0.0)
    dcg = sum((1.0/np.log2(r+2)) for r,idx in enumerate(top5) if idx==gt)
    NDCGs.append(dcg)

print(f"HR@5: {np.mean(HRs):.4f}")
print(f"NDCG@5: {np.mean(NDCGs):.4f}")
