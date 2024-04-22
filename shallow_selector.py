import tensorflow as tf
import scipy
import numpy as np
import copy

class ShallowSelector:
    """
    A class representing a differentially private logistic regression objective function.
    Shallow selection selects top-k items with the largest initial gradient norm || ∂J(~D) / ∂~zi | ~D=D ||.
    """

    def __init__(self, regularization_parameter: float):
        """
        Initialize the ShallowSelector.

        Args:
            regularization_parameter (float): The regularization parameter.
        """
        self.regularization_parameter = regularization_parameter
        self.data_type = tf.float64

    def compute_weights(self, features: np.ndarray, labels: np.ndarray, b=0) -> tf.Tensor:
        """
        Compute the weights for the logistic regression model.

        Args:
            features (np.ndarray): The input features.
            labels (np.ndarray): The labels for the input features.

        Returns:
            np.ndarray: The computed weights.
        """
        # Reconstruct features
        num_samples, num_features = features.shape
        weights = tf.Variable(tf.zeros((num_features,), dtype=self.data_type))
        features_signed = tf.multiply(features, -tf.reshape(labels, (num_samples, 1)))

        optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

        # Optimize log-loss plus L2 term
        def objective_function(b):
            logits = tf.matmul(features_signed, tf.expand_dims(weights, 1))
            loss = tf.reduce_sum(tf.nn.softplus(logits)) \
                    + self.regularization_parameter * tf.reduce_sum(tf.square(weights)) / 2 \
                    + tf.reduce_sum(b * weights)
            return loss

        # train - differentiate weights w.r.t. gradients
        with tf.GradientTape() as tape:
            loss = objective_function(b)
        gradients = tape.gradient(loss, [weights])
        optimizer.apply_gradients(zip(gradients, [weights]))

        return weights

    def shallow_selection(self, features, labels, top_k):
        """
        Perform shallow selection of top-k items with the largest initial gradient norm.

        Args:
            features (np.ndarray): The training features.
            labels (np.ndarray): The labels for the training features.
            top_k (int): The number of top elements to return.

        Returns:
            np.ndarray: The indices of the top k items.
        """
        theta = self.compute_weights(features, labels)
        exponentials = tf.exp(labels * tf.linalg.matvec(features, theta))
        exp_diag = tf.linalg.diag(exponentials / (1 + exponentials) ** 2)
        regularization_matrix_inv = np.linalg.inv(
            self.regularization_parameter * tf.eye(features.shape[1], dtype=self.data_type) +
            tf.linalg.matmul(tf.transpose(features), tf.linalg.matmul(exp_diag, features))
        )
        regularization_matrix_inv[np.isnan(regularization_matrix_inv)] = 0

        grad = tf.TensorArray(self.data_type, size=features.shape[0])
        for j in range(features.shape[0]):
            xj = features[j, :]
            yj = labels[j]
            sj = exponentials[j]
            temp = (yj / (1 + sj)) * tf.eye(features.shape[1], dtype=self.data_type) - \
                    sj * tf.linalg.tensor_diag(theta) @ tf.reshape(xj, (-1, 1)) / (1 + sj) ** 2
            grad = grad.write(j, tf.linalg.matvec(tf.linalg.matmul(temp, regularization_matrix_inv), theta))

        scores = tf.norm(grad.stack(), axis=1)
        return tf.argsort(scores)[-top_k:]

    def attack(self, features, labels, target_points, features_eval, labels_eval, epochs, alpha, eta):
        """
        Perform an attack on a surrogate victim, producing a poisoned training set

        Args:
            features (np.ndarray): The training features.
            labels (np.ndarray): The labels for the training features.
            target_points (np.ndarray): The indices of the items to poison
            features_eval (np.ndarray): The evaluation set features.
            labels_eval (np.ndarray): The evaluation set labels.
            epochs (int): Number of epochs to train over
            alpha (float): Learning rate
            eta (float): Scaling factor
            
        Returns:
            np.ndarray: The indices of the top k items.
        """
        print('Start running label-aversion attack...')

        num_eval_labels = len(labels_eval)

        poison_evolution = {0: features}
        poisoned_features = copy.deepcopy(features)
        for t in range(1, epochs):
            if t % 100 == 0:
                print(t)

            theta = self.compute_weights(poisoned_features, labels).numpy()
            exponentials = tf.exp(-labels_eval * tf.linalg.matvec(features_eval, theta))
            grad_theta = tf.reduce_sum([
                (exponentials[i] / (1 + exponentials[i])) * labels_eval[i] * features_eval[i, :]
                for i in range(num_eval_labels)], axis=0) / num_eval_labels

            # In practice, attack steps needs higher precision than a double
            sigmoid_values = [
                scipy.special.expit(labels[j] * theta.dot(poisoned_features[j, :]))
                for j in range(features.shape[0])
            ]
            sigmoid_diag_matrix = np.diag([sj / np.power(1 + sj, 2) for sj in sigmoid_values])

            # Tensorflow's inverse is less robust than Numpy's and errors out on some datasets
            regularization_matrix_inv = np.linalg.inv(
                self.regularization_parameter * tf.eye(features.shape[1], dtype=self.data_type) +
                tf.linalg.matmul(tf.transpose(features), tf.linalg.matmul(sigmoid_diag_matrix, features))
            )

            # Original approach - performs more poorly than Numpy. Too far into the optimization weeds
            #grad = tf.TensorArray(self.data_type, size=features.shape[0])
            #for j in target_points:
            #    xj = poisoned_features[j, :]
            #    yj = labels[j]
            #    sj = sigmoid_values[j]
            #    temp = (yj / (1 + sj)) * tf.eye(features.shape[1], dtype=self.data_type) - \
            #           sj * tf.linalg.tensor_diag(theta) @ tf.reshape(xj, (-1, 1)) / (1 + sj) ** 2
            #    grad = grad.write(j, tf.linalg.matvec(tf.linalg.matmul(temp, regularization_matrix_inv), theta))
            #poisoned_features -= eta * grad.stack().numpy()
            #....
                
            grad = np.zeros(features.shape)
            for j in target_points:
                xj = np.transpose(poisoned_features[j, :])  # tf transpose seems to struggle during attack on some datasets
                temp = (labels[j] / (1 + sigmoid_values[j])) * np.eye(features.shape[1]) - sigmoid_values[j] * np.outer(theta, xj) / np.power(1 + sigmoid_values[j], 2)
                grad[j, :] = temp.dot(regularization_matrix_inv).dot(grad_theta) + alpha * (xj - features[j, :])

            # Normalize and update
            poisoned_features -= eta * grad
            rownorm = np.sqrt(np.sum(np.abs(poisoned_features)**2, axis=1))
            scale = [[1 if rownorm[i] < 1 else rownorm[i]] for i in range(features.shape[0])]
            poisoned_features /= scale
            poison_evolution[t] = copy.deepcopy(poisoned_features)

        return poison_evolution, poisoned_features
