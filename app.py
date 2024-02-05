import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import streamlit as st

class Predictor:
    def __init__(self, path):
        self.path = path

        if path.endswith(".pickle"):
            self.reg = pickle.load(open(path, "rb"))
        else:
            self.reg = self.train()

    def train(self):
        self.df = pd.read_csv(self.path)
        X = self.df[['# shards (x)', 'total disk usage GB (y)']].to_numpy()
        y = self.df['Memory Footprint GB (z)'].to_numpy()
        return LinearRegression().fit(X, y)

    def calculate_required_specs(self, root_storage_bytes, shard_max_size_bytes=104857600, memory_headroom=2, disk_headroom=1.25):
        est_shards = root_storage_bytes / shard_max_size_bytes
        disk_usage_gb = root_storage_bytes / 1e9
        return dict(
            root_storage_gb=disk_usage_gb,
            required_disk_gb=disk_usage_gb * 2.8 * disk_headroom,
            required_memory_gb=self.estimate_required_memory_gb(est_shards, disk_usage_gb, headroom=memory_headroom),
        )

    def estimate_required_memory_gb(self, num_shards, total_disk_usage_gb, headroom=2, min_gb=0.35):
        return round(max(min_gb, abs(self.reg.predict(np.array([[num_shards, total_disk_usage_gb]])))[0] * headroom), 4)

    def export(self, filename="model.pickle"):
        pickle.dump(self.reg, open(filename, "wb"))
        return filename




p = Predictor("models/zoekt-model.pickle")

st.write("""
# Zoekt Sizing Recommendation

This is an experimental tool to help determine the total resources required to run zoekt nodes that power GitLab's new code search engine.
""")

st.divider()

num_gb = st.slider("Total Unindexed Disk Size GB", 0.1, 1000.0, step=0.1, format="%f GB")
shard_max_gb = st.slider("Shard Max GB (default setting is 0.1 GB)", 0.1, 1.0, step=0.1, format="%f GB")

data = p.calculate_required_specs(num_gb*1e9, shard_max_size_bytes=shard_max_gb*1e9)
required_memory_gb = data['required_memory_gb']
required_disk_gb = data['required_disk_gb']


st.write(f"""
## Recommendation:
* Total Memory: `{required_memory_gb:.2f} GB`
* Total Disk Space: `{required_disk_gb:.2f} GB`
""")

