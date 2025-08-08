from dataclasses import dataclass
from icecream import ic
import jax
import jax.numpy as jnp
from flax import nnx
from einops import einsum
import polars as pl

from harmonic_gains.model import MarketEstimator

df = pl.read_csv("dataset.csv")
dataset = df.select(
    [
        pl.col("date").str.to_datetime().alias("timestamp"),
        pl.col("close").alias("price"),
    ]
).set_sorted("timestamp")

SPLIT = 0.9
split_point = int(SPLIT * len(df))
train_dataset, test_dataset = dataset.slice(0, split_point), dataset.slice(split_point)

print(test_dataset)

rngs = nnx.Rngs(0)


model = MarketEstimator(
    degree=1,
    initial_num_freqs=1024,
    rngs=rngs,
)


def mape(y_true, y_pred):
    return jnp.mean(jnp.abs((y_true - y_pred) / y_true)) * 100


test_prices = test_dataset["price"].to_numpy()[:1000]
test_timestamps = jnp.array(test_dataset["timestamp"].dt.timestamp().to_numpy())[:1000]
reconstructed_prices = model(test_timestamps)
print(reconstructed_prices)
mape_score = mape(test_prices, reconstructed_prices)
ic(mape_score)
