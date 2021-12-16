import torch
from pathlib import Path
path = '/Users/hariharan/hari_works/nonlocal_se/processed_data'
for p in Path(path).iterdir():
    try:
        scan, mask, score = torch.load(str(p))
    except:
        print(p)
        continue
    has_mask = 0
    if mask is None:
        mask = torch.zeros_like(scan)
        has_mask = 1
    torch.save([scan, mask, score, torch.tensor(has_mask)], str(p))