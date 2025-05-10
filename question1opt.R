set.seed(123) # Set seed for reproducibility

# Generate data as specified
n <- 100
x <- rnorm(n, mean = 1, sd = sqrt(2))
y <- rnorm(n, mean = 2 + 3*x, sd = sqrt(5))

# Create design matrix X with intercept
X <- cbind(1, x)

# Function to calculate the loss (MSE)
calculate_loss <- function(beta, X, y) {
  n <- length(y)
  predictions <- X %*% beta
  return(sum((y - predictions)^2) / (2*n))
}

# Function to calculate the gradient
calculate_gradient <- function(beta, X, y) {
  n <- length(y)
  predictions <- X %*% beta
  return(-(t(X) %*% (y - predictions)) / n)
}

# Function to calculate gradient norm
calculate_gradient_norm <- function(gradient) {
  return(sqrt(sum(gradient^2)))
}

# Gradient Descent implementation
gradient_descent <- function(X, y, beta_init, step_size, threshold, max_iter = 10000) {
  beta <- beta_init
  n_iter <- 0
  loss_history <- numeric(max_iter)
  beta_history <- matrix(0, nrow = max_iter, ncol = length(beta_init))
  gradient_norm_history <- numeric(max_iter)
  
  while (n_iter < max_iter) {
    n_iter <- n_iter + 1
    
    # Calculate current loss
    current_loss <- calculate_loss(beta, X, y)
    loss_history[n_iter] <- current_loss
    
    # Calculate gradient
    gradient <- calculate_gradient(beta, X, y)
    
    # Calculate gradient norm
    gradient_norm <- calculate_gradient_norm(gradient)
    gradient_norm_history[n_iter] <- gradient_norm
    
    # Store current beta values
    beta_history[n_iter, ] <- beta
    
    # Check convergence
    if (gradient_norm < threshold) {
      cat("Converged after", n_iter, "iterations\n")
      break
    }
    
    # Update beta
    beta <- beta - step_size * gradient
  }
  
  if (n_iter == max_iter) {
    cat("Maximum iterations reached without convergence\n")
  }
  
  # Trim history vectors to actual iteration count
  loss_history <- loss_history[1:n_iter]
  beta_history <- beta_history[1:n_iter, , drop = FALSE]
  gradient_norm_history <- gradient_norm_history[1:n_iter]
  
  return(list(
    beta = beta,
    loss_history = loss_history,
    beta_history = beta_history,
    n_iter = n_iter,
    gradient_norm_history = gradient_norm_history
  ))
}

# Set parameters for gradient descent
beta_init <- c(0, 0)
step_size <- 0.01
threshold <- 1e-6
max_iter <- 10000

# Run gradient descent
result <- gradient_descent(X, y, beta_init, step_size, threshold, max_iter)

# Print results
cat("Final beta values:", result$beta, "\n")
cat("Final loss value:", tail(result$loss_history, 1), "\n")
cat("Number of iterations:", result$n_iter, "\n")

# Plot loss vs iterations
par(mfrow = c(2, 2))

# Plot 1: Loss vs Iterations
plot(1:result$n_iter, result$loss_history, type = "l", 
     xlab = "Iterations", ylab = "Loss", 
     main = "Loss vs Iterations", col = "blue")

# Plot 2: Log Loss vs Iterations
plot(1:result$n_iter, log(result$loss_history), type = "l", 
     xlab = "Iterations", ylab = "Log Loss", 
     main = "Log Loss vs Iterations", col = "red")

# Plot 3: Descent Path in Parameter Space
plot(result$beta_history[, 1], result$beta_history[, 2], type = "l", 
     xlab = expression(beta[0]), ylab = expression(beta[1]), 
     main = "Descent Path in Parameter Space", col = "purple")
points(result$beta_history[1, 1], result$beta_history[1, 2], col = "red", pch = 16)  # Starting point
points(result$beta_history[result$n_iter, 1], result$beta_history[result$n_iter, 2], col = "green", pch = 16)  # Ending point