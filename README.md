# Physics-informed neural networks for solving differential equations with proportional delay and subtraction delay


In this work, we improve the Physic-informed neural networks(PINNs) to approach ordinary/partial delay
differential equations, including proportional dealy and subtraction delay. In order to deal with the delay
term, we set the trining data into two parts, one part is in delay and the other is not. By making full use
of the modern Auto-Differentiation tools, the neural network fits the solution well after training. Numerical
results obrained using Pytorch in Python illustrate the effciency and the accuracy of this method. Our
method can be extend to handle other delay problems naturally, such as delay initial/boundary conditions
and mixed delay problems.
