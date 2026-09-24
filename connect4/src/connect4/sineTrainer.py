import torch
import math


class LegendrePolynomial3(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. Check out `Extending torch.autograd <https://docs.pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd>`_
        for further details.
        """
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Store input for use in the backward pass using ``ctx.save_for_backward``.
        Other objects can be stored directly as attributes on the ctx object,
        such as ``ctx.my_object = my_object``.
        """
        input, = inputs
        ctx.save_for_backward(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)


dtype = torch.float
device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else "cpu"
)

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Create random Tensors for weights. For this example, we need
# 4 weights: y = a + b * P3(c + d * x), these weights need to be initialized
# not too far from the correct result to ensure convergence.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-6
for t in range(2000):
    # To apply our Function, we use Function.apply method. We alias this as 'P3'.
    P3 = LegendrePolynomial3.apply

    # Forward pass: compute predicted y using operations; we compute
    # P3 using our custom autograd operation.
    y_pred = a + b * P3(c + d * x)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        # Gradients are optional in PyTorch's type definitions, but all four
        # parameters participate in the loss and therefore have gradients.
        assert a.grad is not None
        assert b.grad is not None
        assert c.grad is not None
        assert d.grad is not None
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')


import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import time

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Polynomial Training Visualizer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Polynomial Gradient Descent Visualizer")
st.write(
    "Watch a cubic polynomial learn to approximate "
    r"$y = \sin(x)$ using gradient descent."
)

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

st.sidebar.header("Training Settings")

learning_rate = st.sidebar.number_input(
    "Learning rate",
    min_value=0.00000001,
    max_value=0.001,
    value=0.000001,
    format="%.8f"
)

epochs = st.sidebar.slider(
    "Number of epochs",
    min_value=100,
    max_value=10000,
    value=2000,
    step=100
)

points = st.sidebar.slider(
    "Number of data points",
    min_value=100,
    max_value=5000,
    value=2000,
    step=100
)

speed = st.sidebar.slider(
    "Visualization update interval",
    min_value=1,
    max_value=100,
    value=25,
    help="Update the graphs every N training iterations."
)

# ---------------------------------------------------------
# Generate data
# ---------------------------------------------------------

x = np.linspace(-math.pi, math.pi, points)
y = np.sin(x)

# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "training" not in st.session_state:
    st.session_state.training = False



