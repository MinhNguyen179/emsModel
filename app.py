import os

from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO
from flask import Flask
from tensorflow.keras.optimizers import Adam
import mlflow
import time
from threading import Thread
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app)
loaded_model = load_model('mnist_model.keras')
should_pause = False

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


class MLflowLoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Log training metrics
        mlflow.log_metric("loss", logs.get('loss'), step=epoch)
        mlflow.log_metric("accuracy", logs.get('accuracy'), step=epoch)
        mlflow.log_metric("mean_squared_error", logs.get('mean_squared_error'), step=epoch)

        # Log validation metrics
        if 'val_loss' in logs:
            mlflow.log_metric("val_loss", logs.get('val_loss'), step=epoch)
        if 'val_accuracy' in logs:
            mlflow.log_metric("val_accuracy", logs.get('val_accuracy'), step=epoch)
        if 'val_mean_squared_error' in logs:
            mlflow.log_metric("val_mean_squared_error", logs.get('val_mean_squared_error'), step=epoch)


@socketio.on('start_training')
def start_training(learning_rate, num_epochs, batch_size, session_id):
    # Convert parameters to correct types
    learning_rate = float(learning_rate)
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)

    with mlflow.start_run(run_name=f"Session_{session_id}", nested=True):
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)

        for epoch in range(int(num_epochs)):
            history = loaded_model.fit(
                train_images.reshape(-1, 28, 28, 1), train_labels,
                epochs=num_epochs,
                batch_size=batch_size,
                validation_data=(test_images.reshape(-1, 28, 28, 1), test_labels),
                verbose=1,
                callbacks=[MLflowLoggingCallback()]
            )

            # Extract metrics
            loss = history.history['loss'][0]
            accuracy = history.history['accuracy'][0]
            mean_squared_error = history.history['mean_squared_error'][0]
            val_loss = history.history['val_loss'][0]
            val_accuracy = history.history['val_accuracy'][0]
            val_mean_squared_error = history.history['val_mean_squared_error'][0]

            # Emit progress update
            socketio.emit('training_progress', {
                'epoch': epoch + 1,
                'loss': loss,
                'accuracy': accuracy,
                'mean_squared_error': mean_squared_error,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_mean_squared_error': val_mean_squared_error,
                'session_id': session_id
            })

            # Log the same metrics to MLflow
            mlflow.log_metrics({
                'loss': history.history['loss'][0],
                'accuracy': history.history['accuracy'][0],
            }, step=epoch)
        # Notify completion
        socketio.emit('training_complete', {
            'message': 'Training completed',
            'session_id': session_id
        })


def start_training_thread(learning_rate, num_epochs, batch_size, session_id):
    thread = Thread(target=start_training, args=(learning_rate, num_epochs, batch_size, session_id))
    thread.start()
    return thread


# Function to pause training
@app.route('/pause_training', methods=['POST'])
def pause_training():
    global should_pause
    should_pause = True
    # Log the pause event in MLflow
    with mlflow.start_run():
        mlflow.log_param("training_status", "paused")
    return "Training will be paused after current epoch"


# Endpoint to resume training
@app.route('/resume_training', methods=['POST'])
def resume_training():
    data = request.json
    learning_rate = data.get('learning_rate', 0.001)
    num_epochs = data.get('num_epochs', 5)
    batch_size = data.get('batch_size', 32)
    resume_epoch = data.get('resume_epoch', 0)

    # Log the resume event in MLflow
    with mlflow.start_run():
        mlflow.log_param("training_status", "resumed")
        mlflow.log_param("resumed_from_epoch", resume_epoch)

    # Load model
    model_load_path = f"saved_models/model_epoch_{resume_epoch}.h5"
    if os.path.exists(model_load_path):
        global loaded_model
        loaded_model = load_model(model_load_path)

    # Resume training
    start_training(learning_rate, num_epochs, batch_size, resume_epoch=resume_epoch)
    return send_from_directory('static', 'index.html')


@app.route('/start_training', methods=['POST'])
def start_training_route():
    data = request.json  # Assuming JSON data is sent
    learning_rate = data.get('learning_rate', 0.001)  # Default values as fallback
    num_epochs = data.get('num_epochs', 5)
    batch_size = data.get('batch_size', 32)
    session_id = data.get('session_id')  # Unique identifier for the training session

    # Call the training function with these parameters
    start_training(learning_rate, num_epochs, batch_size, session_id)
    return send_from_directory('static', 'index.html')


@app.route('/')
def main():
    return send_from_directory('static', 'index.html')


if __name__ == '__main__':
    app.run(port=8080, debug=True)
