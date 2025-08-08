import jax
import jax.numpy as jnp
from jax.scipy.special import factorial
from flax import nnx
import datetime as dt


def compute_position_kinematic(t: jax.Array, terms: jax.Array):
    powers = jnp.arange(len(terms))
    factorials = factorial(powers)
    return jnp.sum(terms * (t**powers) / factorials)


class MarketEstimator(nnx.Module):
    def __init__(self, degree: int, initial_num_freqs: int, *, rngs: nnx.Rngs):
        self.degree = degree
        self.initial_num_freqs = initial_num_freqs
        self.rngs = rngs

        self.time_min = dt.datetime(2000, 1, 1).timestamp()
        self.time_max = dt.datetime(2050, 1, 1).timestamp()

        self.freqs = nnx.Param(
            jax.random.uniform(
                self.rngs(), (self.initial_num_freqs,), minval=0.1, maxval=10.0
            )
        )

        self.coefs = nnx.Param(
            jax.random.normal(
                self.rngs(),
                (
                    self.degree,
                    self.initial_num_freqs,
                ),
                dtype=jnp.complex64,
            )
        )

    @nnx.vmap(in_axes=(None, 0))
    def __call__(self, time: jax.Array) -> jax.Array:
        normalized_time = (time - self.time_min) / (self.time_max - self.time_min)

        coefs = jax.vmap(compute_position_kinematic, in_axes=(None, 1))(
            normalized_time, self.coefs.value
        )

        amplitude = jnp.sum(
            coefs * jnp.exp(2j * jnp.pi * self.freqs.value * normalized_time)
        )

        return amplitude.real
