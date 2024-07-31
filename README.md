# Bitcoin Price Prediction Using LSTM

**Project Description:**

This project focuses on predicting Bitcoin prices against the Indian Rupee (INR) using Long Short-Term Memory (LSTM) networks. LSTMs, a type of Recurrent Neural Network (RNN), are particularly effective for time series forecasting due to their ability to retain information over long sequences.

**Libraries Used:**
- **NumPy:** For numerical operations and data manipulation.
- **Pandas:** For data handling and preprocessing.
- **Matplotlib:** For data visualization and plotting.
- **Scikit-Learn:** For data splitting and performance evaluation.
- **TensorFlow/Keras:** For building and training the LSTM model.

**Key Functions and Components:**

- **Data Preprocessing:**
  - **`load_data(file_path)`**: Loads historical Bitcoin price data from a CSV file.
  - **`preprocess_data(df)`**: Cleans and normalizes the data to prepare it for model training.
  - **`create_dataset(data, time_step)`**: Converts the data into sequences for training the LSTM model.

- **Model Building:**
  - **`build_lstm_model(input_shape)`**: Constructs the LSTM model with specified input shape and layers.
  - **`compile_model(model)`**: Compiles the LSTM model with appropriate loss function and optimizer.

- **Training and Evaluation:**
  - **`train_model(model, X_train, y_train, epochs, batch_size)`**: Trains the LSTM model on the training data.
  - **`evaluate_model(model, X_test, y_test)`**: Evaluates model performance using metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE).

- **Visualization:**
  - **`plot_predictions(y_true, y_pred)`**: Plots the predicted prices against actual prices for visual assessment.

**Achievements:**
- Successfully predicted Bitcoin prices with a well-performing LSTM model.
- Achieved a robust evaluation score, demonstrating the model’s reliability in forecasting.

This project showcases proficiency in time series forecasting using LSTM networks and highlights skills in data preprocessing, model building, and performance evaluation.
