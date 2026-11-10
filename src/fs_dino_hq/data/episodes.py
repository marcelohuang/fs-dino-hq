"""Episodic K-shot dataset driven by a pre-computed episode plan.

An episode plan is a CSV file (written by ``fs-dino-hq make-episodes``) with
one row per episode and the columns

    episode_idx, component, query_image, query_json,
    support_1_image, support_1_json, ..., support_K_image, support_K_json

(paths relative to the dataset root; further columns are ignored). Plan order
is the training schedule: the trainers read the rows sequentially and wrap
around, so the episodes seen in each epoch are fixed by the plan and the
epoch number. Letterbox preprocessing runs inside the DataLoader workers.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

from .labelme import labelme_json_to_mask, open_image_rgb
from .letterbox import LetterboxPreprocess, letterbox_mask


class PlanEpisodeDataset(Dataset):
    """Deterministic K-shot episodes from a plan CSV.

    ``offset`` and ``limit`` select the slice of plan rows served (rows wrap
    around the end of the plan); ``limit=0`` serves the whole plan.

    ``hr_mask_size > 0`` adds ``query_mask_hr``: the query mask letterboxed
    onto a canvas of that size with the same top-left convention (uint8). It
    covers the same normalised canvas as ``query_mask``, so a point-sampled
    loss can read it at the coordinates at which it reads the logits.
    """

    def __init__(self, plan_csv: Path, root: Path, n_shot: int,
                 prep: LetterboxPreprocess, label_filter=None,
                 offset: int = 0, limit: int = 0, hr_mask_size: int = 0):
        with open(plan_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows, f"empty plan: {plan_csv}"
        for i in range(1, n_shot + 1):
            assert f"support_{i}_image" in rows[0], \
                f"plan lacks support_{i}_image (plan built for fewer shots?)"
        self.rows, self.root = rows, Path(root)
        self.n_shot, self.prep, self.label_filter = n_shot, prep, label_filter
        self.offset, self.limit = offset, (limit if limit > 0 else len(rows))
        self.hr_mask_size = int(hr_mask_size)

    def __len__(self):
        return min(self.limit, len(self.rows))

    def plan_row(self, idx: int) -> Dict:
        return self.rows[(self.offset + idx) % len(self.rows)]

    def __getitem__(self, idx: int):
        r = self.plan_row(idx)
        episode_idx = int(r.get("episode_idx", idx))
        q_img = open_image_rgb(self.root / r["query_image"])
        q_mask = labelme_json_to_mask(self.root / r["query_json"], self.label_filter)
        sup_d, sup_m = [], []
        for i in range(1, self.n_shot + 1):
            s_img = open_image_rgb(self.root / r[f"support_{i}_image"])
            s_mask = labelme_json_to_mask(self.root / r[f"support_{i}_json"], self.label_filter)
            sup_d.append(self.prep.image_for_dino(s_img))
            sup_m.append(self.prep.mask_for_dino(s_mask))
        item = {
            "support_dino": torch.stack(sup_d, 0),          # [K,3,S,S]
            "support_mask": torch.stack(sup_m, 0),          # [K,1,S,S]
            "query_dino": self.prep.image_for_dino(q_img),  # [3,S,S]
            "query_mask": self.prep.mask_for_dino(q_mask),  # [1,S,S]
            "component": r["component"],
            "episode_idx": episode_idx,
        }
        if self.hr_mask_size > 0:
            item["query_mask_hr"] = letterbox_mask(q_mask, self.hr_mask_size).to(torch.uint8)
        return item


def collate_plan_episodes(items: List[Dict]) -> Dict:
    out = {
        "support_dino": torch.stack([it["support_dino"] for it in items], 0),
        "support_mask": torch.stack([it["support_mask"] for it in items], 0),
        "query_dino": torch.stack([it["query_dino"] for it in items], 0),
        "query_mask": torch.stack([it["query_mask"] for it in items], 0),
        "component": [it["component"] for it in items],
        "episode_idx": [it["episode_idx"] for it in items],
    }
    if "query_mask_hr" in items[0]:
        out["query_mask_hr"] = torch.stack([it["query_mask_hr"] for it in items], 0)
    return out
