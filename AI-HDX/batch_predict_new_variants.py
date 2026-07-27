"""
Batch prediction for new XYN1 variants: E112C, E112I, E112Q, N111D
Uses XYN1.csv peptide list; output saved to XYNmodels/predictions/
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from embedding import seq_embedding

MODELS_DIR   = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/Seq2HDX/models_xyn_updated"
EMBEDDING_DIR = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/embedding"
INPUT_CSV    = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/Seq2HDX/results/XYN/XYN1.csv"
OUTPUT_DIR   = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/predictions"

NEW_VARIANTS = ["E112C", "E112I", "E112Q", "N111D"]

print("Loading models...")
models = [tf.keras.models.load_model(os.path.join(MODELS_DIR, f"NNmodel_m{i}")) for i in range(1, 11)]
colnames = [f"model{i}" for i in range(1, 11)]

confidx = [0.435003467810392, 0.584915875217703, 0.691762575882027,
           0.707836210244849, 0.626804716837474, 0.591011203879881,
           0.532705643129479, 0.426545116775042, 0.251348832172361, 0.36508649008649]

def pred_range(a):
    a = float(np.clip(a, 0.0, 1.0))
    for i in range(0, 10):
        if a > i * 0.1 and a <= (i + 1) * 0.1:
            return confidx[i]
    return confidx[0]

def prediction(prot1, df1):
    x_test = prot1.reshape(prot1.shape[0], 30, 36)
    out_df = df1.reset_index(drop=True)
    for idx, model in enumerate(models):
        y_pred = model.predict(x_test).flatten()
        out_df[colnames[idx]] = y_pred
    out_df["average"] = out_df[colnames].mean(axis=1)
    out_df["SD"] = out_df[colnames].std(axis=1)
    out_df["CI"] = [pred_range(p) for p in out_df["average"]]
    return out_df

for name in NEW_VARIANTS:
    vec_file = os.path.join(EMBEDDING_DIR, f"{name}.vector.txt")
    if not os.path.exists(vec_file):
        print(f"WARNING: {vec_file} not found, skipping.")
        continue
    print(f"Processing: {name}")
    prot, df = seq_embedding(INPUT_CSV, vec_file)
    output_df = prediction(prot, df)
    out_path = os.path.join(OUTPUT_DIR, f"{name}_pred_tuned.csv")
    output_df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

print("\nAll done.")
