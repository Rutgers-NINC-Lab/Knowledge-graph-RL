"""Small artificial inputs for the core algorithm; no original-data access."""
import csv
import json
from pathlib import Path

import numpy as np


def generate(root: Path, config):
    data = root / "data"
    kb = data / "train_data"
    kb.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261104)
    nodes = [f"demo semiology {i:02d}" for i in range(1, 17)]
    regions = [f"Demo SR {i:02d}" for i in range(1, 9)]
    relations = ["has a frequent co-occurrence with", "has a moderate co-occurrence with", "has a rare co-occurrence with"]
    edges = [(node, relations[j % 3], nodes[(i + offset) % 16])
             for i, node in enumerate(nodes) for j, offset in enumerate([1, 2, 4, 7])]

    def csv_file(path, header, rows):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    csv_file(kb / "f2sr_Sheet1.csv", ["Function", "Region"],
             [(f"demo function {i}", regions[i % 8]) for i in range(16)] +
             [(f"demo function {i}", regions[(i + 1) % 8]) for i in [0, 4, 8, 12]])
    csv_file(kb / "brain_function_symptom_mapping1.csv", ["Function", "Semiology"],
             [(f"demo function {i}", n) for i, n in enumerate(nodes)])
    csv_file(kb / "fucn_conn.csv", ["Network", "Region 1", "Region 2", "Weight"],
             [("Demo network", regions[i], regions[(i + 1) % 8], 1) for i in range(8)])
    csv_file(kb / "struc_conn_pairwise.csv", ["Region 1", "Region 2"],
             [(regions[i], regions[(i + 2) % 8]) for i in range(8)])
    csv_file(kb / "final_triples.csv", ["entity1", "relation", "entity2"],
             edges + [(n, "0.9", regions[i % 8]) for i, n in enumerate(nodes)])
    sample = {"synthetic": True, "seed": 20261104, "nodes": nodes, "regions": regions,
              "relations": relations, "edges": edges,
              "temporal_pairs": [[nodes[i], nodes[i + 1], 1.0] for i in range(7)],
              "training_sequences": [[nodes[(i + j) % 16] for j in range(5)] for i in range(8)]}
    (data / "sample.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
    dimension = config["model"]["entity_dim"]
    if dimension < 2 or dimension % 2:
        raise ValueError("The example entity dimension must be a positive even number.")
    entity = rng.normal(0, 0.2, (16, dimension)).astype("float32")
    rotate_entity = rng.normal(0, 0.2, (16, dimension)).astype("float32")
    # The source scorer reads the second half as phase angles.
    phase = rng.uniform(-np.pi, np.pi, (3, dimension // 2)).astype("float32")
    rotate_relation = np.concatenate([np.zeros_like(phase), phase], axis=1)
    np.savez(data / "embeddings.npz", entity=entity, rotate_entity=rotate_entity,
             rotate_relation=rotate_relation)
    # Separate SR references are used only after decoding, never in training or ranking.
    csv_file(data / "references.csv", ["Case_ID", "Start_Semiology", "Path"],
             [(f"SYNTHETIC-{i + 1:02d}", n, " -> ".join(regions[(i + j) % 8] for j in [0, 2, 3, 5]))
              for i, n in enumerate(nodes)])
    csv_file(data / "expert_pairs.csv", ["Case_ID", "Expert_A", "Expert_B"],
             [(f"SYNTHETIC-{i+1:02d}", " -> ".join(regions[(i+j)%8] for j in [0,2,3,5]),
               " -> ".join(regions[(i+j)%8] for j in ([0,2,3] if i%3 == 0 else [0,1,3,5]))) for i in range(16)])
    csv_file(data / "subgroups.csv", ["Start_Semiology", "General_Regions", "Clinical_Type", "Frequency"],
             [(n, "|".join([f"Demo group {i%4+1}"] + ([f"Demo group {(i+1)%4+1}"] if i%3 == 0 else [])),
               f"Demo type {i%3+1}", 1+i%5) for i,n in enumerate(nodes)])


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    generate(root, json.loads((root / "configs/example.json").read_text()))
