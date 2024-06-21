import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_text
from acrobot_used import AcrobotEnv
import pade

# Initialize the Acrobot environment
env = AcrobotEnv()
env.reset()

# Parameters for data collection
dt = 0.02  # Time step
samples = 900  # Number of samples

# Data storage
plot_theta1 = np.zeros(samples)
plot_dtheta1 = np.zeros(samples)
plot_theta2 = np.zeros(samples)
plot_dtheta2 = np.zeros(samples)
plot_torque = np.zeros(samples)

data = np.zeros((samples, 5))
target_theta1 = np.zeros(samples)
target_dtheta1 = np.zeros(samples)
target_theta2 = np.zeros(samples)
target_dtheta2 = np.zeros(samples)

for i in range(samples):
    torque = random.choice(env.AVAIL_TORQUE)
    duration = random.random() * 0.5


    env.tag_state()
    first = True
    # Simulate the environment
    t = duration
    while t > 0:
        env.step(torque)
        t -= dt

        if first:
            (theta10, theta20, dtheta10, dtheta20) = env.get_tagged_state()
            (theta1, theta2, dtheta1, dtheta2) = env.state
            (delta_theta1, delta_theta2, delta_dtheta1, delta_dtheta2) = env.get_tag_diff()

            # Store the data
            plot_theta1[i] = np.degrees(theta1)
            plot_theta2[i] = np.degrees(theta2)
            plot_dtheta1[i] = np.degrees(dtheta1)
            plot_dtheta2[i] = np.degrees(dtheta2)
            plot_torque[i] = torque

            data[i, 0] = theta10
            data[i, 1] = dtheta10
            data[i, 2] = theta20
            data[i, 3] = dtheta20
            data[i, 4] = torque

            target_theta1[i] = delta_theta1
            target_theta2[i] = delta_theta2
            target_dtheta1[i] = delta_dtheta1
            target_dtheta2[i] = delta_dtheta2

            first = False

# Visualize the data
scatter_th1th2 = plt.scatter(plot_theta1, plot_theta2, c=plot_torque, cmap='viridis', vmin=-1, vmax=1)
plt.title("Theta1 vs. Theta2 with Torque Color Map")
plt.colorbar(ticks=[-1, 0, 1])
plt.xticks([-180, -90, 0, 90, 180])
plt.yticks([-180, -90, 0, 90, 180])
plt.xlabel(r'$\theta_1$ (degrees)')
plt.ylabel(r'$\theta_2$ (degrees)')
plt.show()

scatter_th1dth1 = plt.scatter(plot_theta1, plot_dtheta1, c=plot_torque, cmap='viridis', vmin=-1, vmax=1)
plt.title("Theta1 vs. dTheta1 with Torque Color Map")
plt.colorbar(ticks=[-1, 0, 1])
plt.xticks([-180, -90, 0, 90, 180])
plt.yticks([-360, 0, 360])
plt.xlabel(r'$\theta_1$ (degrees)')
plt.ylabel(r'$\dot{\theta}_1$ (degrees/s)')
plt.show()

scatter_th2dth2 = plt.scatter(plot_theta1, plot_dtheta2, c=plot_torque, cmap='viridis', vmin=-1, vmax=1)
plt.title("Theta1 vs. dTheta2 with Torque Color Map")
plt.colorbar(ticks=[-1, 0, 1])
plt.xticks([-180, -90, 0, 90, 180])
plt.yticks([-360, 0, 360])
plt.xlabel(r'$\theta_1$ (degrees)')
plt.ylabel(r'$\dot{\theta}_2$ (degrees/s)')
plt.show()


# Train the qualitative model using PADE
q_table = pade.pade(data, target_dtheta1, nNeighbours=10)
q_labels = pade.create_q_labels(q_table[:, 4:5], ['torque'])

classes, class_names = pade.enumerate_q_labels(q_labels)

classifier = tree.DecisionTreeClassifier(min_impurity_decrease=0.005)
model = classifier.fit(data[:, 0:2], classes)

# Print the decision tree rules for theta 1
tree_rules = export_text(model, feature_names=['theta1', 'dtheta1'])
print(tree_rules)

# Convert threshold values from radians to degrees
for i, threshold in enumerate(model.tree_.threshold):
    if threshold != tree._tree.TREE_UNDEFINED:
        model.tree_.threshold[i] = np.degrees(threshold)

# Visualize the decision tree for theta 1
tree.plot_tree(model, feature_names=['theta1', 'dtheta1'], class_names=class_names, filled=True)
plt.show()