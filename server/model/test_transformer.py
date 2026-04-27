import pytest
import pandas as pd
import numpy as np
from column_transformer import Transformer


class TestBMICategory:
    """Тесты для категорий BMI"""
    
    def test_bmi_underweight(self):
        """BMI < 18.5 → категория 3"""
        df = pd.DataFrame({
            'age': [30],
            'bmi': [17.0],
            'children': [0],
            'sex': ['male'],
            'smoker': ['no']
        })
        result = Transformer().feature_engineering(df)
        assert result['bmi_category'].iloc[0] == 3
    
    def test_bmi_normal(self):
        """18.5 ≤ BMI < 25 → категория 1"""
        df = pd.DataFrame({
            'age': [30],
            'bmi': [22.0],
            'children': [0],
            'sex': ['male'],
            'smoker': ['no']
        })
        result = Transformer().feature_engineering(df)
        assert result['bmi_category'].iloc[0] == 1
    
    def test_bmi_overweight(self):
        """25 ≤ BMI < 30 → категория 7"""
        df = pd.DataFrame({
            'age': [30],
            'bmi': [27.0],
            'children': [0],
            'sex': ['male'],
            'smoker': ['no']
        })
        result = Transformer().feature_engineering(df)
        assert result['bmi_category'].iloc[0] == 7
    
    def test_bmi_obese(self):
        """BMI ≥ 30 → категория 10"""
        df = pd.DataFrame({
            'age': [30],
            'bmi': [35.0],
            'children': [0],
            'sex': ['male'],
            'smoker': ['no']
        })
        result = Transformer().feature_engineering(df)
        assert result['bmi_category'].iloc[0] == 10


class TestAgeCategory:
    """Тесты для категорий возраста"""
    
    def test_age_young(self):
        """age < 22 → категория 1"""
        df = pd.DataFrame({
            'age': [20],
            'bmi': [25.0],
            'children': [0],
            'sex': ['male'],
            'smoker': ['no']
        })
        result = Transformer().feature_engineering(df)
        assert result['age_category'].iloc[0] == 1
    
    def test_age_adult(self):
        """22 ≤ age < 35 → категория 5"""
        df = pd.DataFrame({
            'age': [30],
            'bmi': [25.0],
            'children': [0],
            'sex': ['male'],
            'smoker': ['no']
        })
        result = Transformer().feature_engineering(df)
        assert result['age_category'].iloc[0] == 5
    
    def test_age_middle(self):
        """35 ≤ age < 60 → категория 10"""
        df = pd.DataFrame({
            'age': [45],
            'bmi': [25.0],
            'children': [0],
            'sex': ['male'],
            'smoker': ['no']
        })
        result = Transformer().feature_engineering(df)
        assert result['age_category'].iloc[0] == 10
    
    def test_age_senior(self):
        """age ≥ 60 → категория 15"""
        df = pd.DataFrame({
            'age': [65],
            'bmi': [25.0],
            'children': [0],
            'sex': ['male'],
            'smoker': ['no']
        })
        result = Transformer().feature_engineering(df)
        assert result['age_category'].iloc[0] == 15


class TestFeatureEngineering:
    """Тесты для создания новых признаков"""
    
    def test_creates_bmi_category(self):
        """Проверка: создается ли bmi_category"""
        df = pd.DataFrame({
            'age': [30, 40],
            'bmi': [22.0, 28.0],
            'children': [0, 1],
            'sex': ['male', 'female'],
            'smoker': ['no', 'yes']
        })
        result = Transformer().feature_engineering(df)
        assert 'bmi_category' in result.columns
    
    def test_creates_age_category(self):
        """Проверка: создается ли age_category"""
        df = pd.DataFrame({
            'age': [30, 40],
            'bmi': [22.0, 28.0],
            'children': [0, 1],
            'sex': ['male', 'female'],
            'smoker': ['no', 'yes']
        })
        result = Transformer().feature_engineering(df)
        assert 'age_category' in result.columns
    
    def test_creates_interactions(self):
        """Проверка: создаются ли взаимодействия признаков"""
        df = pd.DataFrame({
            'age': [30, 40],
            'bmi': [22.0, 28.0],
            'children': [0, 1],
            'sex': ['male', 'female'],
            'smoker': ['no', 'yes']
        })
        result = Transformer().feature_engineering(df)
        
        # Должно быть хотя бы одно взаимодействие
        interaction_cols = [col for col in result.columns if '_' in col]
        assert len(interaction_cols) > 0
    
    def test_output_shape(self):
        """Проверка: размер датафрейма увеличивается"""
        df = pd.DataFrame({
            'age': [30, 40, 50],
            'bmi': [22.0, 28.0, 32.0],
            'children': [0, 1, 2],
            'sex': ['male', 'female', 'male'],
            'smoker': ['no', 'yes', 'no']
        })
        
        input_cols = len(df.columns)
        result = Transformer().feature_engineering(df)
        output_cols = len(result.columns)
        
        assert output_cols > input_cols


class TestTransform:
    """Тесты для трансформера"""
    
    def test_transform_returns_array(self):
        """Проверка: transform возвращает numpy array"""
        transformer = Transformer()
        df = pd.DataFrame({
            'age': [30, 40],
            'bmi': [25.0, 28.0],
            'children': [0, 1],
            'sex': ['male', 'female'],
            'smoker': ['no', 'yes']
        })
        
        transformer.fit(df)
        result = transformer.transform(df)
        
        assert isinstance(result, np.ndarray)
    
    def test_transform_shape(self):
        """Проверка: форма результата transform"""
        transformer = Transformer()
        df = pd.DataFrame({
            'age': [30, 40, 50],
            'bmi': [25.0, 28.0, 32.0],
            'children': [0, 1, 2],
            'sex': ['male', 'female', 'male'],
            'smoker': ['no', 'yes', 'no']
        })
        
        transformer.fit(df)
        result = transformer.transform(df)
        
        # 3 строки, 5 колонок (age, bmi, children, sex, smoker)
        assert result.shape == (3, 5)