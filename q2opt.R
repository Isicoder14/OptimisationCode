data <- data.frame(
  x1 = c(-0.829, 0.747294, -0.0209, 1.277665, 0.547097, -0.21768, 0.825416, 1.305479, 
         0.681953, 0.324166, 0.096996, -0.81822, -1.00602, 1.158111, 0.62412, -0.01225, 
         0.075805, 0.97512, -0.8255, 0.412931, -0.82222, 0.244967, -0.47104, -1.44808, 
         -0.71844, 0.310908, 0.85766, -0.01902, -0.01851, 0.322719, 0.519347, -0.10876, 
         0.690144, 0.224092, 0.097676, 0.02451, 1.451144, 2.153182, 0.872321, 2.189803, 
         -0.83972, -2.1239, -0.75913, 0.341756, 0.950424, -0.89841, -1.32023, 1.17944, 
         -1.71313, -0.11454, -1.59443, 0.005244, -0.45007, -1.06762, 0.120296, 0.711615, 
         -1.53411, 0.332314, 1.551152, 1.179297, 2.060748, -0.24896, 0.645376, -0.96492, 
         1.058424, -1.18326, -0.26941, 1.502357, 1.628616, -1.70338, 0.384065, -2.06744, 
         -1.30447, 0.366598, -0.51387, -0.06268, -0.98573, -0.53026, -0.10703, -0.55365, 
         1.964725, -0.69973, -0.11233, 0.614167, -0.5305, -0.27505, -1.51519, 1.644968, 
         0.576557, 3.078881, -0.11792, -1.60645, -0.75635, -0.64657, 1.687142, -0.00797, 
         0.077368, 1.523124, -1.03725, -0.87562),
  x2 = c(-0.56018, 0.61037, 0.117327, -0.59157, -0.20219, 1.098777, 0.81351, 0.021004, 
         -0.31027, -0.13014, 0.595157, 2.092387, -1.21419, 0.791663, 0.628346, -0.89725, 
         -0.67716, -0.14706, -0.32139, -0.56372, 0.243687, -0.50694, 0.23205, -1.40746, 
         -0.21345, 1.475356, -0.15994, -1.00253, -0.28866, -0.82723, 1.532739, 0.401712, 
         -0.40122, 0.012592, -0.77301, 0.497998, 0.959271, -0.76735, 0.183342, -0.8083, 
         -0.59939, -0.52576, 0.150394, 1.876171, -0.5769, 0.491919, 1.831459, -0.46918, 
         1.353872, 1.237816, -0.59938, 0.046981, 0.62285, -0.14238, 0.514439, -1.12464, 
         1.277677, -0.74849, 0.115675, 0.067518, 1.755341, 0.971571, 1.368632, 0.686051, 
         -1.75874, -2.03923, 0.717542, 0.074095, -1.3801, -0.05555, -0.03269, -0.08912, 
         0.669673, -0.93988, -1.05921, 0.955142, 0.504047, -0.79287, -1.03524, -1.19788, 
         0.035264, 0.21398, -0.22097, 0.757508, -0.57582, -2.30192, 1.366874, -0.24904, 
         0.31125, 1.119575, -0.95554, 0.203464, -1.42225, -1.08155, 0.88164, 1.479944, 
         -0.86128, 0.53891, -0.19034, -1.3828),
  y = c(1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 
       1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
       0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 
       0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0)
)

# Function to calculate the sigmoid 
sigmoid <- function(z) {
  return(1 / (1 + exp(-z)))
}

# Function to calculate the negative log-likelihood (objective function)
neg_log_likelihood <- function(beta, X, y) {
  z <- X %*% beta
  # To avoid numerical issues, we'll cap z values
  z[z > 20] <- 20
  z[z < -20] <- -20
  
  # Calculate log-likelihood
  log_likelihood <- sum(log(1 + exp(-y * z)))
  return(log_likelihood)
}

# Function to calculate the gradient of the negative log-likelihood
gradient_neg_log_likelihood <- function(beta, X, y) {
  z <- X %*% beta
  # To avoid numerical issues, we'll cap z values
  z[z > 20] <- 20
  z[z < -20] <- -20
  
  # Calculate gradient
  gradient <- -t(X) %*% (y * (1 - sigmoid(y * z)))
  return(gradient)
}

# Function to perform gradient descent
gradient_descent <- function(X, y, beta_init, learning_rate = 0.05, epsilon = 1e-5, max_iter = 10000) {
  # Initialize parameters
  beta <- beta_init
  converged <- FALSE
  iter <- 0
  loss_history <- numeric(max_iter)
  
  # Add intercept column to X
  X <- cbind(1, X)
  
  while (!converged && iter < max_iter) {
    # Calculate current loss
    current_loss <- neg_log_likelihood(beta, X, y)
    loss_history[iter + 1] <- current_loss
    
    # Calculate gradient
    grad <- gradient_neg_log_likelihood(beta, X, y)
    
    # Update parameters
    beta_new <- beta - learning_rate * grad
    
    # Check convergence
    if (norm(grad, "2") < epsilon) {
      converged <- TRUE
      loss_history <- loss_history[1:(iter + 1)]
    }
    
    beta <- beta_new
    iter <- iter + 1
  }
  
  return(list(
    beta = beta,
    loss_history = loss_history,
    iterations = iter,
    converged = converged,
    final_gradient_norm = norm(gradient_neg_log_likelihood(beta, X, y), "2")
  ))
}

# Prepare the data for the model
X <- as.matrix(data[, c("x1", "x2")])
y <- as.vector(2 * data$y - 1)  # Convert 0,1 to -1,1 for mathematical convenience

# Initial parameter values
beta_init <- matrix(c(0, 0, 0), nrow = 3)  # [intercept, beta1, beta2]

# Run gradient descent
result <- gradient_descent(X, y, beta_init, learning_rate = 0.05, epsilon = 1e-5)

# Print results
cat("Converged:", result$converged, "\n")
cat("Number of iterations:", result$iterations, "\n")
cat("Final parameters (beta):", result$beta, "\n")
cat("Final gradient norm:", result$final_gradient_norm, "\n")
cat("Final loss value:", result$loss_history[length(result$loss_history)], "\n")

# 1. Loss vs iterations(plot)
plot(1:length(result$loss_history), result$loss_history, 
     type = "l", col = "blue", lwd = 2,
     xlab = "Iterations", ylab = "Negative Log-Likelihood",
     main = "Convergence of Gradient Descent")
