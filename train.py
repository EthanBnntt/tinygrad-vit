"""
This file trains the model on MNIST.
"""

from tinygrad import Tensor, nn, TinyJit
from tinygrad.nn.datasets import mnist

from model import ViTModel

X_train, Y_train, X_test, Y_test = mnist()

model = ViTModel(
    image_width=28,
    image_height=28,
    patch_width=7,
    patch_height=7,
    channels=1,
    embed_dim=256,
    hidden_dim=512,
    num_heads=32,
    dropout_p=0.15,
    bias=True,
    num_layers=2,
)

BATCH_SIZE = 256
NUM_BATCHES = 1000
LR = 0.0001

optim = nn.optim.AdamW(nn.state.get_parameters(model), lr=LR)

def step():
    """Perform a single optimization step."""
    Tensor.training = True  # makes dropout work
    samples = Tensor.randint(BATCH_SIZE, high=X_train.shape[0])
    x_train_batch, y_train_batch = X_train[samples], Y_train[samples]

    optim.zero_grad()
    labels, _ = model(x_train_batch.float())
    labels = labels.float()
    y_train_batch = y_train_batch.float()
    loss = labels.sparse_categorical_crossentropy(y_train_batch)
    loss.backward()

    optim.step()
    return loss

jit_step = TinyJit(step)

for step in range(NUM_BATCHES):
    train_loss = jit_step()
    if step % 100 == 0:
        Tensor.training = False
        test_samples = Tensor.randint(BATCH_SIZE, high=X_test.shape[0])
        X_test_batch, Y_test_batch = X_test[test_samples], Y_test[test_samples]
        test_labels, _ = model(X_test_batch.float())
        acc = (test_labels.argmax(axis=1) == Y_test_batch).mean().item()
        print(f"step {step:4d}, loss {train_loss.item():.2f}, acc {acc*100.:.2f}%")
