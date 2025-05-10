data <- c(
  5.363733, 5.496441, 4.081278, 3.300311, 6.660672, 3.287832, 5.143132, 
  4.044685, 5.95796, 5.667324, 7.07508, 3.979967, 4.46025, 3.042473, 
  4.111413, 5.754601, 6.513977, 3.155669, 6.739212, 7.711276, 5.82687, 
  8.753592, 3.452422, 2.510691, 1.44256, 7.992089, 6.308731, 4.888831,
  
  5.559937, 2.749022, 9.891504, 5.258442, 5.21879, 6.451533, 5.962018, 
  5.447768, 3.419051, 5.942937, 8.764049, 7.69084, 8.186373, 3.977569, 
  3.02079, 4.748426, 5.11145, 7.188383, 1.615071, 8.059101, 4.683984, 
  4.146238, 2.975791, 1.690287, 6.646341, 5.146636, 2.420078, 2.409842, 4.328431,
  
  8.338043, 4.480817, 1.993714, 4.508514, 4.454553, -0.39377, 4.89141, 
  4.538131, 6.392413, 8.697912, 7.25313, 4.462223, 2.786948, 10.14672, 
  5.118437, 5.027859, 4.95175, 5.39617, 4.711279, 3.852676, 3.906282, 
  4.934493, 3.91315, 3.574308, 5.21286, 4.490046, 8.007986, -0.30194, 7.183014,
  
  7.49217, 0.85322, 4.314625, 4.257118, 2.184977, 3.444367, 2.778848, 
  8.504541, 6.871357, 7.54311, 6.443344, 2.741896, 3.950959, 5.978749
)

# Define the negative log-likelihood function for normal distribution
neg_log_likelihood <- function(params, data) {
  mu <- params[1]
  sigma <- params[2]
  
  if (sigma <= 0) {
    return(Inf)  # To avoid negative sigma values
  }
  
  n <- length(data)
  nll <- n * log(sigma) + sum((data - mu)^2) / (2 * sigma^2)
  return(nll)
}

# Define the gradient of the negative log-likelihood
gradient_neg_log_likelihood <- function(params, data) {
  mu <- params[1]
  sigma <- params[2]
  
  if (sigma <= 0) {
    return(c(0, 1e6))  # Return large gradient to move away from invalid region
  }
  
  n <- length(data)
  
  # Partial derivative with respect to mu
  d_mu <- -sum(data - mu) / (sigma^2)
  
  # Partial derivative with respect to sigma
  d_sigma <- n / sigma - sum((data - mu)^2) / (sigma^3)
  
  return(c(d_mu, d_sigma))
}

# Gradient Descent Function
gradient_descent <- function(initial_params, data, learning_rate, 
                             epsilon, max_iterations = 10000) {
  
  # Initialize parameters and storage for tracking
  params <- initial_params
  iter <- 0
  converged <- FALSE
  
  # Store history of parameters and function values
  history <- data.frame(
    iteration = integer(),
    mu = numeric(),
    sigma = numeric(),
    nll = numeric(),
    grad_norm = numeric()
  )
  
  while (!converged && iter < max_iterations) {
    # Calculate gradient
    grad <- gradient_neg_log_likelihood(params, data)
    
    # Calculate gradient norm for convergence check
    grad_norm <- sqrt(sum(grad^2))
    
    # Store current values
    current_nll <- neg_log_likelihood(params, data)
    history <- rbind(history, data.frame(
      iteration = iter, 
      mu = params[1], 
      sigma = params[2], 
      nll = current_nll,
      grad_norm = grad_norm
    ))
    
    # Check for convergence
    if (grad_norm < epsilon) {
      converged <- TRUE
      print(paste("Converged after", iter, "iterations"))
    } else {
      # Update parameters
      params <- params - learning_rate * grad
      
      # Ensure sigma remains positive
      params[2] <- max(params[2], 1e-10)
      
      iter <- iter + 1
    }
  }
  
  if (iter == max_iterations) {
    print("Reached maximum iterations without convergence")
  }
  
  # Return the final parameters and history
  return(list(
    params = params,
    history = history,
    converged = converged,
    iterations = iter
  ))
}

# Set parameters
initial_params <- c(0, 1)  # Initial mu and sigma
learning_rate <- 0.01      # Step size (η)
epsilon <- 1e-5            # Convergence threshold (ε)

# Run gradient descent
result <- gradient_descent(initial_params, data, learning_rate, epsilon)

# Print final results
cat("\nFinal Solution:\n")
cat("μ (mean) =", result$params[1], "\n")
cat("σ (std dev) =", result$params[2], "\n")
cat("Negative Log-Likelihood =", neg_log_likelihood(result$params, data), "\n")
cat("Number of Iterations =", result$iterations, "\n")

# Calculate actual mean and standard deviation for comparison
cat("\nActual Statistics from Data:\n")
cat("Sample Mean =", mean(data), "\n")
cat("Sample Standard Deviation =", sd(data), "\n")

# Create convergence plots
par(mfrow = c(2, 2))

# Plot 1: Loss vs. Iterations
plot(result$history$iteration, result$history$nll, 
     type = "l", 
     xlab = "Iterations", 
     ylab = "Negative Log-Likelihood",
     main = "Convergence of Loss Function")

# Plot 2: μ vs. Iterations
plot(result$history$iteration, result$history$mu, 
     type = "l", 
     xlab = "Iterations", 
     ylab = "μ",
     main = "Convergence of μ")

# Plot 3: σ vs. Iterations
plot(result$history$iteration, result$history$sigma, 
     type = "l", 
     xlab = "Iterations", 
     ylab = "σ",
     main = "Convergence of σ")