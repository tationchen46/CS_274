import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops

# ─── Setup & Preprocessing ─────────────────────────────────────────────────────
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = [t for t in text.split() if t not in stop_words]
    tokens = [lemmatizer.lemmatize(stemmer.stem(t)) for t in tokens]
    return ' '.join(tokens)

# ─── Load & Merge Data ─────────────────────────────────────────────────────────
api_df = pd.read_excel('Web API.xls').rename(columns={
    'API名称':'api_name', 'API标签':'labels', 'API服务描述':'description'
})
mashup_df = pd.read_excel('Mashup.xls').rename(columns={
    '名称':'mashup_name', '标签':'labels', '描述':'description',
    '相关API':'used_apis', '类别':'category'
})

# assign node indices
M = len(mashup_df)
api_df['node_index'] = api_df.index + M
mashup_df['node_index'] = mashup_df.index
combined = pd.concat([mashup_df, api_df], ignore_index=True)
N = len(combined)

# parse label sets
def parse_set(s):
    if pd.isna(s) or not s.strip():
        return set()
    return set(x.strip().lower() for x in s.split(',') if x.strip())
combined['label_set'] = combined['labels'].apply(parse_set)

# map mashup → global API indices
api_map = {r.api_name: r.node_index for _, r in api_df.iterrows()}
mashup_used = {}
for _, r in mashup_df.iterrows():
    if pd.isna(r.used_apis) or not r.used_apis.strip():
        mashup_used[r.node_index] = []
    else:
        lst = []
        for name in r.used_apis.split(','):
            name = name.strip()
            if name in api_map:
                lst.append(api_map[name])
        mashup_used[r.node_index] = lst

# preprocess descriptions & TF-IDF
combined['desc_pp'] = combined['description'].fillna('').apply(preprocess_text)
node_vect = TfidfVectorizer(max_features=128, min_df=2, max_df=0.9)
text_vect = TfidfVectorizer(max_features=300, min_df=2, max_df=0.9)

node_feats = torch.tensor(node_vect.fit_transform(combined['desc_pp']).toarray(), dtype=torch.float)
text_feats = torch.tensor(text_vect.fit_transform(combined['desc_pp']).toarray(), dtype=torch.float)

# ─── Build sparse edge_index with boolean mask ────────────────────────────────
mask = np.zeros((N, N), dtype=bool)
for i in range(N):
    for j in range(i+1, N):
        if combined.loc[i, 'label_set'] & combined.loc[j, 'label_set']:
            mask[i, j] = mask[j, i] = True

rows, cols = np.where(mask)
edge_index = torch.tensor([rows, cols], dtype=torch.long)

# add self-loops so SAGEConv includes each node’s own features
edge_index, _ = add_self_loops(edge_index, num_nodes=N)

# ─── Model Definitions ─────────────────────────────────────────────────────────
class SageGCN(nn.Module):
    def __init__(self, num_layers, in_f, hid_f, out_f):
        super().__init__()
        layers = []
        layers.append(SAGEConv(in_f, hid_f))
        for _ in range(num_layers-2):
            layers.append(SAGEConv(hid_f, hid_f))
        layers.append(SAGEConv(hid_f, out_f))
        self.convs = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

class TextEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )
    def forward(self, x):
        return self.enc(x)

def contrastive_loss(a, b, temp=0.3):
    a_n = F.normalize(a, dim=1)
    b_n = F.normalize(b, dim=1)
    logits = (a_n @ b_n.t()) / temp
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)

def bpr_loss(u, p, n):
    pos = (u * p).sum(-1)
    neg = (u * n).sum(-1)
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()

class MSPT_SAGE(nn.Module):
    def __init__(self, cfg, edge_index):
        super().__init__()
        self.sage = SageGCN(
            num_layers=cfg['num_layers'],
            in_f=cfg['node_feat_dim'],
            hid_f=cfg['hidden_dim'],
            out_f=cfg['label_embed_dim']
        )
        self.texter = TextEncoder(
            in_dim=cfg['text_feat_dim'],
            hid_dim=cfg['text_hidden_dim'],
            out_dim=cfg['text_embed_dim']
        )
        self.W4 = nn.Linear(cfg['label_embed_dim'] + cfg['text_embed_dim'],
                            cfg['final_embed_dim'])
        self.edge_index = edge_index.to(cfg['device'])

    def forward(self, node_x, text_x):
        lbl_emb = self.sage(node_x, self.edge_index)
        txt_emb = self.texter(text_x)
        final = self.W4(torch.cat([lbl_emb, txt_emb], dim=1))
        return final, lbl_emb, txt_emb

# ─── Training & Evaluation ─────────────────────────────────────────────────────
config = {
    'num_layers': 2,
    'node_feat_dim': 128,
    'hidden_dim': 64,
    'label_embed_dim': 32,
    'text_feat_dim': 300,
    'text_hidden_dim': 128,
    'text_embed_dim': 32,
    'final_embed_dim': 64,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

device = config['device']
model = MSPT_SAGE(config, edge_index).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

nf = node_feats.to(device)
tf_ = text_feats.to(device)

valid_mashups = [i for i in range(M) if mashup_used.get(i)]
batch_size = 4

# — Pre‑training (contrastive)
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    final, lbl, txt = model(nf, tf_)
    loss_c = contrastive_loss(lbl, txt)
    loss_c.backward()
    optimizer.step()
    print(f"Pretrain {epoch+1}/30  Loss: {loss_c.item():.4f}")

# — Fine‑tuning (BPR)
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    final, _, _ = model(nf, tf_)
    mashups = (random.sample(valid_mashups, batch_size)
               if len(valid_mashups) >= batch_size else valid_mashups)
    u_list, p_list, n_list = [], [], []
    for m in mashups:
        pos = random.choice(mashup_used[m])
        neg_candidates = list(set(range(M, M+len(api_df))) - set(mashup_used[m]))
        neg = random.choice(neg_candidates)
        u_list.append(final[m])
        p_list.append(final[pos])
        n_list.append(final[neg])
    if not u_list:
        continue
    u_b = torch.stack(u_list)
    p_b = torch.stack(p_list)
    n_b = torch.stack(n_list)
    loss_b = bpr_loss(u_b, p_b, n_b)
    loss_b.backward()
    optimizer.step()
    print(f"Finetune {epoch+1}/30  Loss: {loss_b.item():.4f}")

# — Evaluation (HR@5 & NDCG@5)
with torch.no_grad():
    model.eval()
    final, _, _ = model(nf, tf_)
    users = final[:M]
    items = final[M:]
    hr_list, ndcg_list = [], []

    for i in range(M):
        gt = set(mashup_used.get(i, []))
        if not gt:
            continue
        scores = (users[i:i+1] @ items.t()).squeeze()
        top5 = torch.topk(scores, k=5).indices + M

        # Hit Rate
        hr = int(bool(set(top5.tolist()) & gt))
        hr_list.append(hr)

        # NDCG
        dcg = sum(1 / np.log2(r + 2)
                  for r, idx in enumerate(top5.tolist()) if idx in gt)
        ideal = sum(1 / np.log2(r + 2)
                    for r in range(min(len(gt), 5)))
        ndcg_list.append(dcg / ideal if ideal > 0 else 0.0)

    print(f"HR@5: {np.mean(hr_list):.4f}")
    print(f"NDCG@5: {np.mean(ndcg_list):.4f}")
