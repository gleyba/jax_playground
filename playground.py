import time
import jax
import jax.numpy as jnp
from jax import random
from jax import grad, jit
import numpy as np

@jax.jit
def add_one(x: jax.Array) -> jax.Array:
    print("I'm being traced!")
    y = x + 1
    return y


print("First call")
print("Result:", add_one(jnp.array([0.0])))

print("\nSubsequent call with array of the same shape")
print("Result:", add_one(jnp.array([0.0])))

print("\nSubsequent call with array of different shape")
print("Result:", add_one(jnp.array([1.0, 2.0, 3.0])))

@jax.jit
def naive_relu(x: float) -> float:
    if x > 0:
        return x
    else:
        return 0

try:
    naive_relu(10)
except Exception as ex:
    print(f"{type(ex).__name__}")


class TimeMeasure(object):
     def __init__(self, description):
         self.description = description

     def __enter__(self):
         self.start = time.time()
         return self 

     def __exit__(self, exc_type, exc_value, traceback):
        end = time.time()
        print(f"{self.description}: {end - self.start:.4f} seconds")
 
# This example is from https://medium.com/nlplanet/a-quick-intro-to-jax-with-examples-c6e8cc65c3c1

key = random.PRNGKey(0)
size = 5000
x = random.normal(key, (size, size), dtype=jnp.float32)

with TimeMeasure("runs on CPU - JAX"):
    np.dot(x, x.T)

with TimeMeasure("measure JAX device transfer time"):
    x_jax = jax.device_put(x)

with TimeMeasure("runs on GPU - measure JAX with compilation time"):
    jnp.dot(x_jax, x_jax.T).block_until_ready()

with TimeMeasure("runs on GPU - measure JAX running time"):
    jnp.dot(x_jax, x_jax.T).block_until_ready() 
