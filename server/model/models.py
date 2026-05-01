import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

from column_transformer import Transformer
from network import Network

def train_model(model, train_loader, val_loader, scaler_y, epochs=200, patience=30):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping на эпохе {epoch}")
                break
        
        if epoch % 20 == 0:
            # Показываем loss в нормализованном масштабе (он маленький)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    model.load_state_dict(torch.load('best_model.pth'))
    return train_losses, val_losses

if __name__ == "__main__":
    # 1. ЗАГРУЗКА ДАННЫХ
    print("Загрузка данных...")
    data = pd.read_csv("data/insurance.csv")
    train_set, test_set = train_test_split(data, test_size=0.1, random_state=42)
    train_y, test_y = train_set["charges"], test_set["charges"]

    print(f"Train: {train_set.shape}, Test: {test_set.shape}")

    # 2. ТРАНСФОРМАЦИЯ
    print("\nТрансформация данных...")
    transformer = Transformer()
    
    train_set = train_set.drop(["region"], axis=1)
    test_set = test_set.drop(["region"], axis=1)
    
    train_transformed = transformer.fit_transform(train_set)
    test_transformed = transformer.transform(test_set)
    
    train_set[transformer.num_features + transformer.cat_features] = train_transformed
    test_set[transformer.num_features + transformer.cat_features] = test_transformed
    
    train_set = transformer.feature_engineering(train_set)
    test_set = transformer.feature_engineering(test_set)
    
    # Сохраняем признаки
    train_set = train_set.drop(["charges"], axis=1)
    test_set = test_set.drop(["charges"], axis=1)
    
    # 3. RANDOM FOREST (не требует нормализации target)
    print("\n" + "="*50)
    print("RANDOM FOREST")
    print("="*50)
    
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=4, min_samples_split=2)
    rf.fit(train_set, train_y)
    rf_pred = rf.predict(test_set)
    rf_mae = mean_absolute_error(test_y, rf_pred)
    rf_r2 = r2_score(test_y, rf_pred)
    
    print(f"Random Forest MAE: ${rf_mae:.2f}")
    print(f"Random Forest R2: {rf_r2:.4f}")
    
    # 4. НЕЙРОННАЯ СЕТЬ
    print("\n" + "="*50)
    print("НЕЙРОННАЯ СЕТЬ")
    print("="*50)
    
    # Нормализация признаков
    scaler_X = StandardScaler()
    train_scaled = scaler_X.fit_transform(train_set)
    test_scaled = scaler_X.transform(test_set)
    
    # Нормализация целевой переменной (ВАЖНО!)
    scaler_y = StandardScaler()
    train_y_scaled = scaler_y.fit_transform(train_y.values.reshape(-1, 1)).flatten()
    test_y_scaled = scaler_y.transform(test_y.values.reshape(-1, 1)).flatten()
    
    print(f"Target scale: mean={scaler_y.mean_[0]:.2f}, std={scaler_y.scale_[0]:.2f}")
    
    # Тензоры
    train_t = torch.FloatTensor(train_scaled)
    train_y_t = torch.FloatTensor(train_y_scaled)
    test_t = torch.FloatTensor(test_scaled)
    test_y_t = torch.FloatTensor(test_y_scaled)
    
    # DataLoaders
    train_loader = DataLoader(TensorDataset(train_t, train_y_t), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_t, test_y_t), batch_size=32)
    
    # Создаем сеть
    network = Network(input_dim=train_scaled.shape[1], hidden_dims=[128, 64, 32], dropout_rate=0.3)
    print(f"Входных признаков: {train_scaled.shape[1]}")
    print(f"Параметров сети: {sum(p.numel() for p in network.parameters()):,}")
    
    # Обучение
    train_losses, val_losses = train_model(network, train_loader, test_loader, scaler_y, epochs=200, patience=30)
    
    # График обучения
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (normalized)')
    plt.legend()
    plt.title('Обучение нейронной сети')
    
    # Оценка нейросети на тесте
    network.eval()
    with torch.no_grad():
        nn_pred_scaled = network(test_t).numpy()
    
    # Обратное масштабирование предсказаний
    nn_pred = scaler_y.inverse_transform(nn_pred_scaled.reshape(-1, 1)).flatten()
    
    nn_mae = mean_absolute_error(test_y, nn_pred)
    nn_r2 = r2_score(test_y, nn_pred)
    
    print(f"\nNeural Network MAE: ${nn_mae:.2f}")
    print(f"Neural Network R2: {nn_r2:.4f}")
    
    # График предсказаний
    plt.subplot(1, 2, 2)
    plt.scatter(test_y, nn_pred, alpha=0.5, label='Neural Network')
    plt.scatter(test_y, rf_pred, alpha=0.3, label='Random Forest')
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', label='Ideal')
    plt.xlabel('True Charges')
    plt.ylabel('Predicted Charges')
    plt.legend()
    plt.title('Сравнение моделей')
    
    plt.tight_layout()
    plt.show()
    
    # 5. СРАВНЕНИЕ
    print("\n" + "="*50)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*50)
    print(f"Random Forest  -> MAE: ${rf_mae:.2f}, R2: {rf_r2:.4f}")
    print(f"Neural Network -> MAE: ${nn_mae:.2f}, R2: {nn_r2:.4f}")
    
    # 6. СОХРАНЕНИЕ
    print("\nСохранение моделей...")
    joblib.dump(rf, "server/model/best_model.joblib")
    joblib.dump(transformer, "server/model/transformer.joblib")
    joblib.dump(scaler_X, "server/model/scaler_X.joblib")
    joblib.dump(scaler_y, "server/model/scaler_y.joblib")
    torch.save(network.state_dict(), "server/model/neural_network.pth")
    
    print("✅ Все модели сохранены!")