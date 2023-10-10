import random
import matplotlib.pyplot as plt

# Parameters
mean = 0.2
sigma = 0.1
max_noise = 0.4

# Generate random samples
samples = []

while len(samples) < 10000:
    random_noise = random.gauss(mean, sigma)
    if 0 <= random_noise <= max_noise:
        samples.append(random_noise)

# Create a histogram plot
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g', label='Sampled Data')
plt.title('Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Show the plot
# plt.show()
plt.savefig("noise_distribution.png")
