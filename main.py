import jax
import jax.numpy as jnp
from flax import nnx
import optax
import polars as pl
from harmonic_gains.model import MarketEstimator

df = pl.read_csv("dataset.csv")
dataset = df.select(
    [
        pl.col("date").str.to_datetime().alias("timestamp"),
        pl.col("close").alias("price"),
    ]
).set_sorted("timestamp")

dataset = dataset.slice(1000)
split_point = int(0.9 * len(dataset))
train_dataset, test_dataset = dataset.slice(0, split_point), dataset.slice(split_point)

train_prices = train_dataset["price"].to_numpy()
train_timestamps = jnp.array(train_dataset["timestamp"].dt.timestamp().to_numpy())
test_prices = test_dataset["price"].to_numpy()
test_timestamps = jnp.array(test_dataset["timestamp"].dt.timestamp().to_numpy())

rngs = nnx.Rngs(0)
model = MarketEstimator(degree=1, initial_num_freqs=1024, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(0.00001), wrt=nnx.Param)


def mse(y_true, y_pred):
    return jnp.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return jnp.mean(jnp.abs(y_true - y_pred))


def mape(y_true, y_pred):
    return jnp.mean(jnp.abs((y_true - y_pred) / y_true)) * 100


@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        pred = model(x)
        return mse(y, pred), pred

    (loss, pred), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, mae(y, pred), mape(y, pred)


BATCH_SIZE = 2048
step = 0

while True:
    indices = jax.random.choice(rngs(), len(train_prices), (BATCH_SIZE,), replace=True)
    batch_x = train_timestamps[indices]
    batch_y = train_prices[indices]

    train_loss, train_mae, train_mape = train_step(model, optimizer, batch_x, batch_y)

    if step % 100 == 0:
        print(f"MAE: {train_mae:.4f}, MAPE: {train_mape:.2f}%")

    step += 1
