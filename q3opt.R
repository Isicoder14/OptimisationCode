quadratic_func <- function(x, A, b) { # Define the quadratic function
  return(t(x) %*% A %*% x + t(b) %*% x)
}
quadratic_gradient <- function(x, A, b) {
  return(2 * A %*% x + b)
}

# Implement gradient descent
gradient_descent <- function(func, grad_func, x0, A, b, eta, epsilon, max_iter = 10000) {
  x <- x0
  iter <- 0
  converged <- FALSE
  
  # Store values for plotting
  x_history <- matrix(0, nrow = max_iter + 1, ncol = length(x0))
  x_history[1, ] <- x0
  f_history <- numeric(max_iter + 1)
  f_history[1] <- func(x0, A, b)
  grad_norm_history <- numeric(max_iter + 1)
  
  # Calculate initial gradient
  gradient <- grad_func(x, A, b)
  grad_norm_history[1] <- sqrt(sum(gradient^2))
  
  while (iter < max_iter && !converged) {
    iter <- iter + 1
    
    # Update x
    gradient <- grad_func(x, A, b)
    x <- x - eta * gradient
    
    # Store current point and function value
    x_history[iter + 1, ] <- x
    f_history[iter + 1] <- func(x, A, b)
    
    # Calculate gradient norm
    grad_norm <- sqrt(sum(gradient^2))
    grad_norm_history[iter + 1] <- grad_norm
    
    # Check convergence
    if (grad_norm < epsilon) {
      converged <- TRUE
      cat("Converged after", iter, "iterations with gradient norm:", grad_norm, "\n")
    }
    
    # Print progress every 10 iterations
    if (iter %% 10 == 0) {
      cat("Iteration:", iter, "| Function value:", func(x, A, b), "| Gradient norm:", grad_norm, "\n")
    }
  }
  
  return(list(
    x = x,
    f_value = func(x, A, b),
    iterations = iter,
    converged = converged,
    x_history = x_history[1:(iter + 1), ],
    f_history = f_history[1:(iter + 1)],
    grad_norm_history = grad_norm_history[1:(iter + 1)]
  ))
}

# Set up the problem parameters
A <- matrix(c(2, 0, 0, 4), nrow = 2, ncol = 2)
b <- matrix(c(-4, -8), nrow = 2)
x0 <- matrix(c(1, 1), nrow = 2)
eta <- 0.1
epsilon <- 1e-6

# Run gradient descent
result <- gradient_descent(quadratic_func, quadratic_gradient, x0, A, b, eta, epsilon)

# Print results
cat("Final solution: x =", result$x[1], "y =", result$x[2], "\n")
cat("Function value at solution:", result$f_value, "\n")
cat("Number of iterations:", result$iterations, "\n")
cat("Convergence status:", ifelse(result$converged, "Converged", "Did not converge"), "\n")

# Create plots in a 2x2 layout to visualize all results
par(mfrow = c(2, 2))

# Create a plot of the convergence (function value vs. iterations)
plot(0:result$iterations, result$f_history, type = "l", col = "blue", lwd = 2,
     xlab = "Iterations", ylab = "Function Value", main = "Convergence of Gradient Descent")

# Create a plot for gradient norm vs. iterations
plot(0:result$iterations, result$grad_norm_history, type = "l", col = "red", lwd = 2,
     xlab = "Iterations", ylab = "Gradient Norm (log scale)", main = "Gradient Norm vs. Iterations", log = "y")

# Create a separate plot for the trajectory of x and y values
plot(0:result$iterations, result$x_history[, 1], type = "l", col = "darkgreen", lwd = 2,
     xlab = "Iterations", ylab = "Parameter Value", main = "Parameter Trajectory")
lines(0:result$iterations, result$x_history[, 2], col = "purple", lwd = 2)
legend("right", legend = c("x", "y"), col = c("darkgreen", "purple"), lty = 1, cex = 0.8)

# Create a contour plot to visualize the function and the descent path
# Adjust range to ensure we capture the full path and the minimum
x_min <- min(result$x_history[, 1]) - 0.5
x_max <- max(result$x_history[, 1]) + 0.5
y_min <- min(result$x_history[, 2]) - 0.5
y_max <- max(result$x_history[, 2]) + 0.5

x_range <- seq(x_min, x_max, length.out = 100)
y_range <- seq(y_min, y_max, length.out = 100)
z_matrix <- matrix(0, nrow = length(x_range), ncol = length(y_range))

for (i in 1:length(x_range)) {
  for (j in 1:length(y_range)) {
    point <- matrix(c(x_range[i], y_range[j]), nrow = 2)
    z_matrix[i, j] <- quadratic_func(point, A, b)
  }
}

# Create the contour plot
contour(x_range, y_range, z_matrix, nlevels = 20, 
        xlab = "x", ylab = "y", main = "Contour Plot with Descent Path")

lines(result$x_history[, 1], result$x_history[, 2], col = "red", lwd = 2)
points(result$x_history[, 1], result$x_history[, 2], col = "red", pch = 19, cex = 0.7)

# mark start and end points
points(x0[1], x0[2], col = "green", pch = 19, cex = 2)  # Starting point
points(result$x[1], result$x[2], col = "blue", pch = 19, cex = 2)  # Final point
legend("topright", legend = c("Start", "Path", "End"), 
       col = c("green", "red", "blue"), pch = c(19, 19, 19), cex = 0.8)
