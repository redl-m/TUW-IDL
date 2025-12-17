#!/usr/bin/env python

"""
Solution for Problem Sheet 4: Minimal Autograd Engine
192.151 Introduction to Deep Learning
"""

import numpy as np


#
# Problem 4.1: Base Node
#

class Node():
    """
    Base class for a node in the computation graph.
    """

    def __init__(self, data, _parents=()):
        """
        Initializes a Node instance. [cite: 35]

        [cite: 36, 37, 38, 39, 40]
        """
        # Hold data in a np.array(), initialize grad to the same shape as
        # data with values of 0, and keep track of parents in a set

        self.data = np.array(data)  # [cite: 24, 25, 41]

        self.grad = np.zeros_like(self.data)  # [cite: 27, 28, 42]

        self.parents = set(_parents)  # [cite: 30, 43]

        # This will hold the function that computes the gradient
        # for the inputs of this node.
        # It's a no-op by default (for input nodes).
        self.backward_fn = lambda: None

    def __repr__(self):
        """
        Provides a string representation of the node for printing. [cite: 46]
        """

        return f"Node(data=({self.data}), grad=({self.grad}))"  # [cite: 47]

    def backward(self):
        """
        Performs backpropagation starting from this node. [cite: 51]
        """

        backward_list = []  # [cite: 52]
        visited = set()

        # Placeholder for the gradient of the loss (this node is the loss)

        self.grad = np.ones_like(self.data)  # [cite: 54]

        # Define a recursive function to build the backward list (topological sort)

        def build_backward_list(n):  # [cite: 55]
            # Check if the node is in the backward list (via visited set)

            if n not in visited:  # [cite: 56]
                visited.add(n)
                # iterate through the parents of the node and call the
                # function recursively [cite: 57]
                for parent in n.parents:
                    build_backward_list(parent)
                # Add the node to the backward list

                backward_list.append(n)  # [cite: 56]

        build_backward_list(self)  # [cite: 58]

        # Iterate through the backward list (in reverse topological order)
        # and call each backward_fn [cite: 60]
        for node in reversed(backward_list):
            node.backward_fn()


#
# Helper Primitives (from Problem 4.4)
# These are needed to build the activation functions.
#


def log_node(a_node):  # [cite: 147]
    """Natural logarithm node."""

    out = Node(np.log(a_node.data), (a_node,))  # [cite: 149]

    def backward_fn():  # [cite: 150]
        # dL/da = dL/dout * dout/da = dL/dout * (1/a)

        a_node.grad += out.grad / a_node.data  # [cite: 151]

    out.backward_fn = backward_fn  # [cite: 152]

    return out  # [cite: 153]


def exp_node(a_node):  # [cite: 154]
    """Exponential (e^x) node."""

    out = Node(np.exp(a_node.data), (a_node,))  # [cite: 156]

    def backward_fn():  # [cite: 157]
        # dL/da = dL/dout * dout/da = dL/dout * e^a
        # Note: out.data = e^a
        # PDF typo [cite: 158] corrected from 'a_node.grad + out.grad out.data'
        a_node.grad += out.grad * out.data

    out.backward_fn = backward_fn  # [cite: 159]

    return out  # [cite: 160]


def div_node(a_node, b_node):  # [cite: 194]
    """Division node (a / b)."""

    out = Node(a_node.data / b_node.data, (a_node, b_node))  # [cite: 196]

    def backward_fn():  # [cite: 197]
        # dL/da = dL/dout * (1/b)

        a_node.grad += out.grad / b_node.data  # [cite: 198]
        # dL/db = dL/dout * (-a / b^2)

        b_node.grad += -out.grad * a_node.data / (b_node.data ** 2)  # [cite: 199]

    out.backward_fn = backward_fn  # [cite: 200]

    return out  # [cite: 201]


#
# Problem 4.2: Addition and Multiplication Operations
#

def add_node(a_node, b_node):  # [cite: 74]
    """
    Element-wise addition: A + B [cite: 75]
    """
    # Produce a new node:
    # Add the input data together to produce the output for the out node. [cite: 79]
    # Make sure the output knows what its parents are [cite: 81]
    out = Node(a_node.data + b_node.data, (a_node, b_node))

    # define a backward_fn to update a_node's gradients and b_node's gradients
    # based on out.grad [cite: 88, 89]

    def add_backward():  # [cite: 90]
        # dL/da = dL/dc * dc/da = dL/dc * 1
        # Use += to accumulate gradients [cite: 19]
        a_node.grad += out.grad
        # dL/db = dL/dc * dc/db = dL/dc * 1
        b_node.grad += out.grad

    out.backward_fn = add_backward  # [cite: 92, 93]

    return out  # [cite: 94]


def mul_node(a_node, b_node):  # [cite: 105]
    """
    Element-wise multiplication: A * B [cite: 107]
    """
    # Produce a new node:
    # Make sure the output's data is correct [cite: 112]
    # Make sure the output knows what its parents are [cite: 117]
    out = Node(a_node.data * b_node.data, (a_node, b_node))

    # define a backward_fn to update a_node's gradients and b_node's gradients
    # based on out.grad [cite: 121, 122]
    def mul_backward():
        # dL/da = dL/dc * dc/da = dL/dc * b
        a_node.grad += out.grad * b_node.data
        # dL/db = dL/dc * dc/db = dL/dc * a
        b_node.grad += out.grad * a_node.data

    out.backward_fn = mul_backward

    return out  # [cite: 123]


#
# Problem 4.3: Matrix Multiplication
#


def matmul_node(a_node, b_node):  # [cite: 131]

    """Matrix multiplication: A @ B"""  # [cite: 132]
    # Use @ for matrix multiplication in NumPy [cite: 129]

    out = Node(a_node.data @ b_node.data, (a_node, b_node))  # [cite: 134]

    def matmul_backward():
        # Gradients for matrix multiplication [cite: 127]
        # dL/da = dL/dc @ b.T
        a_node.grad += out.grad @ b_node.data.T
        # dL/db = a.T @ dL/dc
        b_node.grad += a_node.data.T @ out.grad

    out.backward_fn = matmul_backward

    return out  # [cite: 135]


#
# Problem 4.4: Activation Functions
#


def softplus_node(x):  # [cite: 161]
    """
    SoftPlus activation: f(x) = ln(1 + e^x) [cite: 142, 163]
    Built by composing primitive nodes. [cite: 138]
    """
    # Create a node for '1' [cite: 144, 145]
    ones = Node(np.ones_like(x.data))

    # e^x

    exp_x = exp_node(x)  # [cite: 146]

    # 1 + e^x
    one_plus_exp_x = add_node(ones, exp_x)

    # ln(1 + e^x)

    out = log_node(one_plus_exp_x)  # [cite: 146]

    return out  # [cite: 166]


def leaky_relu_node(input_node, alpha=0.01):  # [cite: 174]
    """
    Leaky ReLU activation: [cite: 175]
    f(x) = x, if x > 0
    f(x) = alpha*x, otherwise [cite: 176, 177]

    Implemented as a new primitive based on hints. [cite: 171-173, 182-184]
    """
    # Forward pass [cite: 178]
    # Use np.where to apply the condition [cite: 182]
    out_data = np.where(input_node.data > 0,
                        input_node.data,
                        input_node.data * alpha)

    out = Node(out_data, (input_node,))  # [cite: 185]

    def leaky_relu_backward():
        # The derivative is 1 if x > 0, and alpha otherwise.
        # We use x >= 0 to match the test case for x=0. [cite: 332]
        grad_x = np.where(input_node.data >= 0, 1.0, alpha)

        # dL/dx = dL/dout * dout/dx
        input_node.grad += out.grad * grad_x

    out.backward_fn = leaky_relu_backward

    return out  # [cite: 186]


def tanh_node(x):  # [cite: 202]
    """
    Tanh activation: f(x) = (e^x - e^-x) / (e^x + e^-x) [cite: 188, 204]
    Built by composing primitive nodes.
    """
    # Following hints to create separate intermediate nodes [cite: 192, 193, 207]

    # Create a constant node for -1

    minus_one = Node(-np.ones_like(x.data))  # [cite: 206]

    # --- Build Numerator: (e^x - e^-x) ---
    neg_x_num = mul_node(x, minus_one)
    exp_x_num = exp_node(x)
    exp_neg_x_num = exp_node(neg_x_num)

    # For (e^x - e^-x), we do (e^x + (-1 * e^-x))
    sub_term = mul_node(exp_neg_x_num, minus_one)
    numerator = add_node(exp_x_num, sub_term)

    # --- Build Denominator: (e^x + e^-x) ---
    # Create separate branch as per hint [cite: 193]
    neg_x_den = mul_node(x, minus_one)
    exp_x_den = exp_node(x)
    exp_neg_x_den = exp_node(neg_x_den)

    denominator = add_node(exp_x_den, exp_neg_x_den)

    # --- Final Division ---

    out = div_node(numerator, denominator)  # [cite: 194, 208]

    return out


#
# Appendix A: Test Code
# [cite: 210]
#

if __name__ == "__main__":
    print("--- Testing Problem 4.1: Base Node ---")

    # --- Base Node Tests --- [cite: 211]

    print("Testing Node Initialization...")  # [cite: 212]

    a_test = Node(3.0)  # [cite: 213, 215]

    b_test = Node(2.0)  # [cite: 214, 215]

    assert a_test.data == 3.0, "Error: a_test.data not set correctly"  # [cite: 216, 221]

    assert b_test.data == 2.0, "Error: b_test.data not set correctly"  # [cite: 216, 221]

    assert a_test.grad == 0.0, "Error: a_test.grad not set correctly"  # [cite: 216, 221]

    assert b_test.grad == 0.0, "Error: b_test.grad not set correctly"  # [cite: 216, 221]

    print("Passed!")  # [cite: 217]

    # Create graph f = (a+b)*b

    c_test = add_node(a_test, b_test)  # [cite: 222]

    f_test = mul_node(c_test, b_test)  # [cite: 223]

    print("\nTesting Node BACKWARD...")  # [cite: 224]
    # f_test.grad = 1.0 # [cite: 225] This line is redundant, backward() sets it

    f_test.backward()  # [cite: 226]

    print(f"a_grad: {a_test}")  # [cite: 227, 228]

    print(f"b_grad: {b_test}")  # [cite: 229, 230]

    print(f"c_grad: {c_test}")  # [cite: 231, 232]

    print(f"f_grad: {f_test}")  # [cite: 233, 234]

    # --- Backward Test --- [cite: 235]
    # check whether update parent gradients correctly [cite: 236]
    # f = (a+b)*b = (3+2)*2 = 10
    # df/da = b = 2
    # df/db = (a+b)*1 + b*1 = a+2b = 3+2*2 = 7
    # df/dc = b = 2
    # df/df = 1

    assert a_test.grad == 2.0, "Error: BACKWARD does not update gradients correctly"  # [cite: 237]

    assert b_test.grad == 7.0, "Error: BACKWARD does not update gradients correctly"  # [cite: 237]

    assert c_test.grad == 2.0, "Error: BACKWARD does not update gradients correctly"  # [cite: 237]

    assert f_test.grad == 1.0, "Error: BACKWARD does not update gradients correctly"  # [cite: 237]

    print("All tests passed!")  # [cite: 238]
    print("-" * 40)

    #
    # --- Testing Problem 4.2: Operations ---
    #

    # --- Add Operation Tests --- [cite: 242]

    print("Testing Node Addition...")  # [cite: 243]

    a_test = Node(3.0)  # [cite: 240]

    b_test = Node(2.0)  # [cite: 241]

    c_test = add_node(a_test, b_test)  # [cite: 244]

    assert c_test.data == 5.0, "Error: add_node DATA not evaluated correctly"  # [cite: 245]

    assert c_test.grad == 0.0, "Error: add_node GRAD not initialized correctly"  # [cite: 245]

    print("Passed!")  # [cite: 245]

    print("Testing Node Addition Backward...")  # [cite: 246]

    c_test.grad = 1.0  # [cite: 247]

    c_test.backward_fn()  # [cite: 248]

    assert a_test.grad == 1.0, "Error: add_node BACKWARD does not update parent gradients correctly"  # [cite: 249, 251]

    assert b_test.grad == 1.0, "Error: add_node BACKWARD does not update parent gradients correctly"  # [cite: 252, 254]

    print("Passed!")  # [cite: 256]

    # --- Multiplication Operation Tests --- [cite: 259]

    print("\nTesting Node Multiplication...")  # [cite: 262]

    a_test = Node(3.0)  # [cite: 258]

    b_test = Node(2.0)  # [cite: 260]

    d_test = mul_node(a_test, b_test)  # [cite: 263]

    assert d_test.data == 6.0, "Error: mul_node DATA not evaluated correctly"  # [cite: 264, 267]

    assert d_test.grad == 0.0, "Error: mul_node GRAD not initialized correctly"  # [cite: 265, 268]

    print("Passed!")  # [cite: 266]

    print("Testing Node Multiplication Backward...")  # [cite: 269]

    d_test.grad = 1.0  # [cite: 270]

    d_test.backward_fn()  # [cite: 271]

    assert a_test.grad == 2.0, "Error: mul_node BACKWARD does not update parent gradients correctly"  # [cite: 272, 273]

    assert b_test.grad == 3.0, "Error: mul_node BACKWARD does not update parent gradients correctly"  # [cite: 274]

    print("Passed!")  # [cite: 275]

    print("All tests passed!")  # [cite: 276]
    print("-" * 40)

    #
    # --- Testing Problem 4.3: Matrix Multiplication ---
    #

    # --- Tests matmul function --- [cite: 282]

    print("Testing Matrix Multiplication...")  # [cite: 283]

    a_test = Node(np.array([[1, 2], [3, 4]]))  # [cite: 277, 281]

    b_test = Node(np.array([[2, 0], [1, 2]]))  # [cite: 279, 281]

    c_test = matmul_node(a_test, b_test)  # [cite: 284, 285]

    # Expected: [[1*2+2*1, 1*0+2*2], [3*2+4*1, 3*0+4*2]] = [[4, 4], [10, 8]]

    assert np.allclose(c_test.data,
                       np.array([[4, 4], [10, 8]])), "Error: matmul DATA not evaluated correctly"  # [cite: 286, 287]

    print("Passed!")  # [cite: 288]

    print("Testing Matrix Multiplication Backward...")  # [cite: 289]

    c_test.grad = np.array([[1, 1], [1, 1]])  # [cite: 290]

    c_test.backward_fn()  # [cite: 291]

    # dL/da = dL/dc @ b.T = [[1,1],[1,1]] @ [[2,1],[0,2]] = [[2,3],[2,3]]

    assert np.allclose(a_test.grad, np.array(
        [[2, 3], [2, 3]])), "Error: matmul BACKWARD does not update parent gradients correctly"  # [cite: 292, 293]
    # dL/db = a.T @ dL/dc = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]

    assert np.allclose(b_test.grad, np.array(
        [[4, 4], [6, 6]])), "Error: matmul BACKWARD does not update parent gradients correctly"  # [cite: 294, 295]

    print("Passed!")  # [cite: 296]

    print("All tests passed!")  # [cite: 297]
    print("-" * 40)

    #
    # --- Testing Problem 4.4: Activation Functions ---
    #

    # --- Tests for SoftPlus --- [cite: 298]
    print("Testing SoftPlus...")

    x = Node(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))  # [cite: 301]

    out = softplus_node(x)  # [cite: 302]

    expected_forward = np.log(1.0 + np.exp(x.data))  # [cite: 304]

    print("Forward Output (softplus(x)):", out.data)  # [cite: 306]

    print("Expected:", expected_forward)  # [cite: 306]

    print("Match:", np.allclose(out.data, expected_forward))  # [cite: 308]
    assert np.allclose(out.data, expected_forward), "Softplus forward failed"

    out.grad = np.ones_like(out.data)  # [cite: 309, 310]

    out.backward()  # [cite: 311]

    expected_grad_x = 1.0 / (1.0 + np.exp(-x.data))  # Sigmoid function [cite: 312]

    print("\nGradient wrt Input:", x.grad)  # [cite: 314]

    print("Expected Gradient:", expected_grad_x)  # [cite: 315]

    print("Match:", np.allclose(x.grad, expected_grad_x))  # [cite: 316]
    assert np.allclose(x.grad, expected_grad_x), "Softplus backward failed"
    print("Passed!")

    # --- Tests for Leaky ReLU --- [cite: 318]
    print("\nTesting Leaky ReLU...")

    x = Node(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))  # [cite: 320]

    alpha = 0.01  # [cite: 321]

    out = leaky_relu_node(x, alpha)  # [cite: 324]

    out.grad = np.ones_like(out.data)  # [cite: 325, 326]

    out.backward()  # [cite: 328]

    expected_forward = np.array([-0.02, -0.01, 0.0, 1.0, 2.0])  # [cite: 331]

    expected_grad_x = np.array([alpha, alpha, 1.0, 1.0, 1.0])  # [cite: 332]

    print("Forward Output:", out.data)  # [cite: 334]

    print("Expected:", expected_forward)  # [cite: 335]

    print("Match:", np.allclose(out.data, expected_forward))  # [cite: 336]
    assert np.allclose(out.data, expected_forward), "Leaky ReLU forward failed"

    print("\nGradient wrt Input:", x.grad)  # [cite: 337]

    print("Expected Gradient:", expected_grad_x)  # [cite: 338]

    print("Match:", np.allclose(x.grad, expected_grad_x))  # [cite: 339]
    assert np.allclose(x.grad, expected_grad_x), "Leaky ReLU backward failed"
    print("Passed!")

    # --- Tests for Tanh --- [cite: 341]
    print("\nTesting Tanh...")

    x = Node(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))  # [cite: 343]

    out = tanh_node(x)  # [cite: 346]

    expected_forward = np.tanh(x.data)  # [cite: 347]

    out.grad = np.ones_like(out.data)  # [cite: 348]

    out.backward()  # [cite: 350]

    # Expected gradients: d/dx tanh(x) = 1 - tanh^2(x) [cite: 352]

    expected_grad_x = 1.0 - expected_forward ** 2  # [cite: 352]

    print("Forward Output:", out.data)  # [cite: 355]

    print("Expected:", expected_forward)  # [cite: 356]

    print("Match:", np.allclose(out.data, expected_forward))  # [cite: 358]
    assert np.allclose(out.data, expected_forward), "Tanh forward failed"

    print("\nGradient wrt Input:", x.grad)  # [cite: 359]

    print("Expected Gradient:", expected_grad_x)  # [cite: 360]

    print("Match:", np.allclose(x.grad, expected_grad_x))  # [cite: 361]
    assert np.allclose(x.grad, expected_grad_x), "Tanh backward failed"
    print("Passed!")

    print("\n" + "=" * 40)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("=" * 40)
