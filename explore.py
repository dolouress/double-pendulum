import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_text
from acrobot_used import AcrobotEnv
import pade

# Initialize the Acrobot environment
env = AcrobotEnv(render_mode='human')
env.reset()

# Parameters for data collection
dt = 0.02  # Time step
samples = 900  # Number of samples

# Data storage
plot_theta1 = np.zeros(samples)
plot_theta2 = np.zeros(samples)
plot_torque = np.zeros(samples)

data = np.zeros((samples, 3))
target_dtheta1 = np.zeros(samples)
target_dtheta2 = np.zeros(samples)

for i in range(samples):
    torque = random.choice(env.AVAIL_TORQUE)
    duration = random.random() * 0.5

    # Tag initial state
    state = env.state
    # print(state)
    theta1, theta2, dtheta1, dtheta2 = state
    env.step(torque)

    # Simulate the environment
    t = duration
    while t > 0:
        env.step(torque)
        t -= dt

    # Capture the final state and state changes
    new_state = env.state
    delta_theta1 = new_state[0] - theta1
    delta_theta2 = new_state[1] - theta2
    delta_dtheta1 = new_state[2] - dtheta1
    delta_dtheta2 = new_state[3] - dtheta2

    # Store the data
    plot_theta1[i] = np.degrees(theta1)
    plot_theta2[i] = np.degrees(theta2)
    plot_torque[i] = torque

    data[i, 0] = theta1
    data[i, 1] = theta2
    data[i, 2] = torque
    target_dtheta1[i] = delta_dtheta1
    target_dtheta2[i] = delta_dtheta2

# Visualize the data
scatter = plt.scatter(plot_theta1, plot_theta2, c=plot_torque, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(ticks=[-1, 0, 1])
plt.show()

# Train the qualitative model using PADE
q_table = pade.pade(data, target_dtheta1, nNeighbours=10)
q_labels = pade.create_q_labels(q_table[:, 2:3], ['torque'])
classes, class_names = pade.enumerate_q_labels(q_labels)

classifier = tree.DecisionTreeClassifier(min_impurity_decrease=0.05)
model = classifier.fit(data[:, 0:2], classes)

# Print the decision tree rules
tree_rules = export_text(model, feature_names=['theta1', 'theta2'])
print(tree_rules)

# Convert threshold values from radians to degrees
for i, threshold in enumerate(model.tree_.threshold):
    if threshold != tree._tree.TREE_UNDEFINED:
        model.tree_.threshold[i] = np.degrees(threshold)

# Visualize the decision tree
tree.plot_tree(model, feature_names=['theta1', 'theta2'], class_names=class_names, filled=True)
plt.show()
