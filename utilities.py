import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix

def generate_uniform_unit_hypersphere(n, d):
    """
    Used to generate training data
    """
    X = np.random.uniform(-1, 1, size=(n, d)).astype(np.float64)
    norms = np.linalg.norm(X, axis=1)
    mask = (norms <= 1)
    X = X[mask]
    while X.shape[0] < n:
        extra_points = np.random.uniform(-1, 1, size=(n - X.shape[0], d))
        extra_norms = np.linalg.norm(extra_points, axis=1)
        extra_mask = extra_norms <= 1
        X = np.vstack((X, extra_points[extra_mask]))
    return X[:n]

def generate_grid_unit_hypersphere(gap, d):
    """
    Used to generate test data
    """
    ndim = int(2.0 / gap + 1)
    grid_dim = np.linspace(-1, 1, ndim).astype(np.float64)
    meshgrids = np.meshgrid(*(grid_dim,) * d, indexing='ij')
    distances = np.sqrt(np.sum(np.square(meshgrid) for meshgrid in meshgrids))
    mask = distances <= 1
    grid = np.column_stack([meshgrid[mask] for meshgrid in meshgrids])
    return grid

def create_model(input_shape):
    """
    Off-the-shelf Keras classifier
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='softmax')  # Output layer with softmax activation for multiclass classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def compute_sum_distance_from_all_other_points(true_dataset, same_label, different_label):
    """
    Supports stand-in label-aversion heuristic
    """
    res = {}
    for i, p1 in enumerate(true_dataset):
        if p1 not in same_label:
            continue
        for p2 in different_label:
            dist = np.sqrt(np.sum((p1-p2)**2))
            if i in res:
                res[i] += dist
            else:
                res[i] = dist

    tmp_k = None
    tmp_v = 0
    for k, v in res.items():
        if v >= tmp_v:
            tmp_v = v
            tmp_k = k
        
    return (tmp_k, tmp_v)

def apply_negation(X_train, y_train, selected_indices):
    """
    A label-aversion-like heuristic designed to produce interim data posionings
    """
    mangled_x = np.copy(X_train)
    mangled_y = np.copy(y_train)
    for unique_label in np.unique(mangled_y):
        same_label = mangled_x[(mangled_y == unique_label)]
        different_label = mangled_x[(mangled_y != unique_label)]
        target_value = compute_sum_distance_from_all_other_points(mangled_x, different_label, same_label)
        print(f"Target value for label {unique_label} is {mangled_x[target_value[0]]}")
    
        for idx in selected_indices:
            if mangled_y[idx] != unique_label:
                continue
            mangled_x[idx] = mangled_x[target_value[0]]
    return mangled_x, mangled_y

def evaluate(X_train, y_train, X_test, y_test):
    """
    Evaluate model performance given (poisoned) training and (unpoisoned) test data
    """
    model = create_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=False)
    prediction_prob = model.predict(X_test)  # Probabilities for each class
    prediction = tf.argmax(prediction_prob, axis=1).numpy()  # Convert to discrete labels (0, 1, 2, or 3)
    prediction[np.isnan(prediction)] = 0

    # Calculation was inspired by https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    cnf_matrix = confusion_matrix(y_test, prediction)
    false_pos = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    false_neg = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    true_pos = np.diag(cnf_matrix)
    true_neg = cnf_matrix.sum() - (false_pos + false_neg + true_pos)
    
    precision = true_pos / (true_pos + false_pos)
    accuracy = (true_pos + true_neg) / (true_pos + false_pos + false_neg + true_neg)
    
    print(f"Precision: {np.mean(PPV)}, Accuracy: {np.mean(accuracy)}")
    print(f"Times correct: {np.count_nonzero(prediction == y_test)/len(X_test)}")
    return np.mean(PPV), np.mean(accuracy)
