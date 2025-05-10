rosenbrock_func <- function(x) {
  return((1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
}

rosenbrock_gradient <- function(x) {
  grad_x <- -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
  grad_y <- 200 * (x[2] - x[1]^2)
  return(c(grad_x, grad_y))
}

gradient_descent <- function(func, grad_func, x0, eta, epsilon, max_iter = 10000) {
  x <- x0
  iter <- 0
  converged <- FALSE
  
  # Stores values for plotting
  x_history <- matrix(0, nrow = max_iter + 1, ncol = length(x0))
  x_history[1, ] <- x0
  f_history <- numeric(max_iter + 1)
  f_history[1] <- func(x0)
  grad_norm_history <- numeric(max_iter + 1)
  
  # Initial gradient
  gradient <- grad_func(x)
  grad_norm_history[1] <- sqrt(sum(gradient^2))
  
  while (iter < max_iter && !converged) {
    iter <- iter + 1
    
    # Update x
    gradient <- grad_func(x)
    x <- x - eta * gradient
    
    # Store current point and function value
    x_history[iter + 1, ] <- x
    f_history[iter + 1] <- func(x)
    
    # Calculate gradient norm
    grad_norm <- sqrt(sum(gradient^2))
    grad_norm_history[iter + 1] <- grad_norm
    
    # Check convergence
    if (grad_norm < epsilon) {
      converged <- TRUE
      cat("Converged after", iter, "iterations with gradient norm:", grad_norm, "\n")
    }
    
    #To see progress every 1000 iterations
    if (iter %% 1000 == 0) {
      cat("Iteration:", iter, "| Function value:", func(x), "| Gradient norm:", grad_norm, "\n")
    }
  }
  
  return(list(
    x = x,
    f_value = func(x),
    iterations = iter,
    converged = converged,
    x_history = x_history[1:(iter + 1), ],
    f_history = f_history[1:(iter + 1)],
    grad_norm_history = grad_norm_history[1:(iter + 1)]
  ))
}

x0 <- c(-1, 1)  # Initial point
eta <- 0.001    # Step size
epsilon <- 1e-6  # Convergence threshold

# Run gradient descent
set.seed(123)  # For reproducibility
result <- gradient_descent(rosenbrock_func, rosenbrock_gradient, x0, eta, epsilon)

# Print results
cat("\n--- Final Results ---\n")
cat("Final solution: x =", result$x[1], "y =", result$x[2], "\n")
cat("Function value at solution:", result$f_value, "\n")
cat("Number of iterations:", result$iterations, "\n")
cat("Convergence status:", ifelse(result$converged, "Converged", "Did not converge"), "\n")

# Create plots in a 2x2 layout to visualize all results
par(mfrow = c(2, 2))

# Plot 1: Function value vs. iterations
plot(0:result$iterations, result$f_history, type = "l", col = "blue", lwd = 2,
     xlab = "Iterations", ylab = "Function Value", 
     main = "Convergence of Gradient Descent (Rosenbrock)")

# Plot 2: Gradient norm vs. iterations (log scale)
plot(0:result$iterations, result$grad_norm_history, type = "l", col = "red", lwd = 2,
     xlab = "Iterations", ylab = "Gradient Norm", log = "y",
     main = "Gradient Norm vs. Iterations (log scale)")

# Plot 3: Path trajectory (x and y values)
plot(0:result$iterations, result$x_history[, 1], type = "l", col = "darkgreen", lwd = 2,
     xlab = "Iterations", ylab = "Parameter Value", main = "Parameter Trajectory")
lines(0:result$iterations, result$x_history[, 2], col = "purple", lwd = 2)
legend("right", legend = c("x", "y"), col = c("darkgreen", "purple"), lty = 1, cex = 0.8)

# Plot 4: Contour plot of the Rosenbrock function with descent path
# Create a grid of points
x_range <- seq(min(min(result$x_history[, 1]) - 0.5, -1.5), max(max(result$x_history[, 1]) + 0.5, 1.5), length.out = 100)
y_range <- seq(min(min(result$x_history[, 2]) - 0.5, -0.5), max(max(result$x_history[, 2]) + 0.5, 1.5), length.out = 100)
z_matrix <- matrix(0, nrow = length(x_range), ncol = length(y_range))

# Calculate function values at grid points
for (i in 1:length(x_range)) {
  for (j in 1:length(y_range)) {
    z_matrix[i, j] <- rosenbrock_func(c(x_range[i], y_range[j]))
  }
}

# To create contour plot
contour(x_range, y_range, z_matrix, nlevels = 20, 
        xlab = "x", ylab = "y", main = "Contour Plot with Descent Path",
        drawlabels = FALSE)

lines(result$x_history[, 1], result$x_history[, 2], col = "red", lwd = 2)
points(result$x_history[seq(1, nrow(result$x_history), 
                            max(1, floor(nrow(result$x_history)/20))), 1], 
       result$x_history[seq(1, nrow(result$x_history), 
                            max(1, floor(nrow(result$x_history)/20))), 2], 
       col = "red", pch = 19, cex = 0.7)

#mark start and end points
points(x0[1], x0[2], col = "green", pch = 19, cex = 2)  # Starting point
points(result$x[1], result$x[2], col = "blue", pch = 19, cex = 2)  # Final point
legend("topright", legend = c("Start", "Path", "End"), 
       col = c("green", "red", "blue"), pch = c(19, 19, 19), cex = 0.8)