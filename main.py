from icecream import ic
import jax
import jax.numpy as jnp
from flax import nnx
from einops import einsum
import polars as pl

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

INITIAL_PARAMS = 64

freqs = jax.random.uniform(rngs(), (INITIAL_PARAMS,), minval=0.1, maxval=10.0)
complex_coefs = jax.random.normal(rngs(), (INITIAL_PARAMS,), dtype=jnp.complex64)


def estimate_prices(timestamps, freqs, coefs):
    time_normalized = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])

    signal = einsum(
        coefs,
        jnp.exp(2j * jnp.pi * freqs[:, None] * time_normalized[None, :]),
        "i, i j -> j",
    )

    reconstructed_prices = signal.real
    return reconstructed_prices


def mape(y_true, y_pred):
    return jnp.mean(jnp.abs((y_true - y_pred) / y_true)) * 100


test_prices = test_dataset["price"].to_numpy()
test_timestamps = test_dataset["timestamp"].dt.timestamp().to_numpy()

reconstructed_prices = estimate_prices(test_timestamps, freqs, complex_coefs)
mape_score = mape(test_prices, reconstructed_prices)

ic(f"MAPE on test dataset: {mape_score:.2f}%")
ic(f"Original prices shape: {test_prices.shape}")
ic(f"Reconstructed prices shape: {reconstructed_prices.shape}")
