# Physics-informed neural networks for solving differential equations with proportional delay and subtraction delay


In this work, we improve the Physic-informed neural networks(PINNs) to approach ordinary/partial delay
differential equations, including proportional delay and subtraction delay. In order to deal with the delay
term, we set the training data into two parts, one part is in delay and the other is not. By making full use
of the modern Auto-Differentiation tools, the neural network fits the solution well after training. Numerical
results obrained using Pytorch in Python illustrate the effciency and the accuracy of this method. Our
method can be extend to handle other delay problems naturally, such as delay initial/boundary conditions
and mixed delay problems.


## description and usage:
1."PINN-ODE... .py" gives the network and figures of example 1 in section 3.1.
2."PINN-PDE with proportional delay.py" and "Load_Models_pde_proportional_delay.py" are codes for example 2 in section3.2 .
3."PINN-PDE with subtraction delay.py" and "Load_Models_pde_subtraction_delay.py" are used in section3.3 for example3,
  "series solution.csv"  gives the value of "truncated series solution" in grid points by  mathematica 12.0, this file will
  will be used while comparing with PINN method in related "Load_Model" code.
  
Readers should run  the PINN source code first and the "Load..." file later in each example.
