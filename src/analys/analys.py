import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Analys:
    """Класс для визуализации и анализа данных"""
    
    @staticmethod
    def scatter_target_corr(df: pd.DataFrame, corr_m=None, target_col='charges'):
        """
        Создает scatter plots для целевой переменной и каждого признака
        
        Parameters:
        -----------
        df : pd.DataFrame
            Данные
        corr_m : pd.DataFrame, optional
            Матрица корреляций
        target_col : str
            Имя целевой переменной
            
        Returns:
        --------
        fig, axes
            Объекты matplotlib
        """
        if corr_m is None:
            # Используем только числовые колонки для корреляции
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_m = df[numeric_cols].corr()
        
        # Получаем все признаки кроме целевой
        features = [col for col in corr_m.columns if col != target_col]
        n_features = len(features)
        
        if n_features == 0:
            print("Нет признаков для отображения!")
            return None, None
        
        # Создаем сетку графиков
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(n_cols*5, n_rows*4))
        
        # Преобразуем axes в плоский массив
        if n_features == 1:
            axes = np.array([axes])
        
        if hasattr(axes, 'flat'):
            axes_flat = list(axes.flat)
        else:
            axes_flat = [axes]
        
        # Получаем корреляции с целевой переменной
        if target_col in corr_m.columns:
            target_correlations = corr_m[target_col]
        else:
            target_correlations = pd.Series()
        
        # Сортируем признаки по абсолютной корреляции
        sorted_features = sorted(features, 
                                key=lambda x: abs(target_correlations.get(x, 0)), 
                                reverse=True)
        
        print(f"Отображаем {len(sorted_features)} признаков против {target_col}")
        
        for idx, feature in enumerate(sorted_features):
            if idx < len(axes_flat):
                axe = axes_flat[idx]
                
                # Проверяем, что признак существует
                if feature not in df.columns:
                    print(f"Признак '{feature}' отсутствует в DataFrame")
                    continue
                
                # Scatter plot
                axe.scatter(df[feature], df[target_col], 
                          alpha=0.5, s=30, edgecolor='none')
                
                # Линия регрессии
                try:
                    valid_data = df[[feature, target_col]].dropna()
                    if len(valid_data) > 1:
                        z = np.polyfit(valid_data[feature], valid_data[target_col], 1)
                        p = np.poly1d(z)
                        x_sorted = np.sort(valid_data[feature])
                        axe.plot(x_sorted, p(x_sorted), 
                               "r--", alpha=0.8, linewidth=2)
                except Exception as e:
                    print(f"Ошибка при построении линии регрессии для {feature}: {e}")
                
                # Добавляем коэффициент корреляции
                corr = target_correlations.get(feature, np.nan)
                if not np.isnan(corr):
                    axe.text(0.05, 0.95, f'r={corr:.3f}', 
                           transform=axe.transAxes,
                           fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                
                # Названия осей
                axe.set_xlabel(feature, fontsize=11)
                axe.set_ylabel(target_col, fontsize=11)
                axe.set_title(f'{feature} vs {target_col}', fontsize=12, pad=10)
                
                # Сетка
                axe.grid(True, alpha=0.3, linestyle='--')
        
        # Скрываем пустые подграфики
        for idx in range(len(sorted_features), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.suptitle(f"Scatter Plots: Features vs Target ({target_col})", 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        return fig, axes
    
    @staticmethod
    def plot_correlation_matrix(df, figsize=(10, 8)):
        """Визуализирует матрицу корреляций"""
        # Используем только числовые колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Heatmap
        im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Подписи
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # Добавляем значения в ячейки
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", 
                              color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
        
        # Цветовая шкала
        plt.colorbar(im)
        ax.set_title("Correlation Matrix", fontsize=16, pad=20)
        
        plt.tight_layout()
        return fig, ax