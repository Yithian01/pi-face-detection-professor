import os
import math
from sklearn import neighbors
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

def train_and_save_model(train_dir, model_dir="./models", n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Train a KNN model and save it to the specified directory.

    :param train_dir: Directory with subdirectories for each known person.
    :param model_dir: Directory to save the trained KNN model file.
    :param n_neighbors: Number of neighbors for KNN. If None, automatically chosen.
    :param knn_algo: Algorithm to compute nearest neighbors. Default is 'ball_tree'.
    :param verbose: If True, prints training progress.
    """
    # Prepare training data
    X = []
    y = []

    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                if verbose:
                    print(f"Skipping image {img_path}: Invalid number of faces")
            else:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Automatically determine the number of neighbors if not specified
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print(f"Automatically chose n_neighbors: {n_neighbors}")

    # Train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Save the trained KNN classifier to the specified directory
    model_save_path = os.path.join(model_dir, "trained_knn_model.clf")
    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)

    print(f"Model saved to {model_save_path}")

# Example usage
if __name__ == "__main__":
    train_dir = "./train"  # Path to training data
    model_dir = "./model"  # Path to save the model
    train_and_save_model(train_dir, model_dir=model_dir, n_neighbors=2, verbose=True)