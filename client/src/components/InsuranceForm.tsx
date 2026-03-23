
import React, { useState, type FormEvent } from 'react';
import type { ChangeEvent } from 'react';
import type { CustomerData, InsuranceRequest, ValidationErrors } from '../types/insurance';
import { useInsurance } from '../hooks/useInsurance';
import './InsuranceForm.css';

const InsuranceForm: React.FC = () => {
  const { loading, error, price, calculatePrice, reset: resetApiState } = useInsurance();
  
  const [formData, setFormData] = useState<CustomerData>({
    age: '',
    sex: 'male',
    bmi: '',
    children: '',
    smoker: 'no',
    // region удален
  });

  const [validationErrors, setValidationErrors] = useState<ValidationErrors>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});

  // Валидация формы
  const validateForm = (): boolean => {
    const errors: ValidationErrors = {};
    let isValid = true;

    // Валидация возраста
    const ageNum = Number(formData.age);
    if (!formData.age) {
      errors.age = 'Возраст обязателен';
      isValid = false;
    } else if (isNaN(ageNum)) {
      errors.age = 'Возраст должен быть числом';
      isValid = false;
    } else if (ageNum < 18) {
      errors.age = 'Возраст должен быть не менее 18 лет';
      isValid = false;
    } else if (ageNum > 100) {
      errors.age = 'Возраст должен быть не более 100 лет';
      isValid = false;
    }

    // Валидация BMI
    const bmiNum = Number(formData.bmi);
    if (!formData.bmi) {
      errors.bmi = 'BMI обязателен';
      isValid = false;
    } else if (isNaN(bmiNum)) {
      errors.bmi = 'BMI должен быть числом';
      isValid = false;
    } else if (bmiNum < 10) {
      errors.bmi = 'BMI должен быть не менее 10';
      isValid = false;
    } else if (bmiNum > 50) {
      errors.bmi = 'BMI должен быть не более 50';
      isValid = false;
    }

    // Валидация количества детей
    const childrenNum = Number(formData.children);
    if (formData.children === '' || formData.children === undefined) {
      errors.children = 'Количество детей обязательно';
      isValid = false;
    } else if (isNaN(childrenNum)) {
      errors.children = 'Количество детей должно быть числом';
      isValid = false;
    } else if (childrenNum < 0) {
      errors.children = 'Количество детей не может быть отрицательным';
      isValid = false;
    } else if (childrenNum > 10) {
      errors.children = 'Количество детей не может превышать 10';
      isValid = false;
    }

    setValidationErrors(errors);
    return isValid;
  };

  // Обработка изменения полей
  const handleChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    
    if (type === 'checkbox') {
      const checked = (e.target as HTMLInputElement).checked;
      setFormData(prev => ({ ...prev, [name]: checked ? 'yes' : 'no' }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
    
    // Очищаем ошибки при изменении поля
    if (validationErrors[name as keyof ValidationErrors]) {
      setValidationErrors(prev => ({ ...prev, [name]: undefined }));
    }
    
    // Сбрасываем состояние API при изменении данных
    resetApiState();
  };

  // Обработка потери фокуса
  const handleBlur = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name } = e.target;
    setTouched(prev => ({ ...prev, [name]: true }));
    validateForm();
  };

  // Отправка формы
  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    
    // Отмечаем все поля как touched для показа ошибок
    const allTouched: Record<string, boolean> = {};
    Object.keys(formData).forEach(key => {
      allTouched[key] = true;
    });
    setTouched(allTouched);
    
    if (!validateForm()) {
      return;
    }

    // Подготовка данных для API
    const requestData: InsuranceRequest = {
      age: Number(formData.age),
      sex: formData.sex,
      bmi: Number(formData.bmi),
      children: Number(formData.children),
      smoker: formData.smoker,
      // region удален
    };

    // Вызываем calculatePrice
    calculatePrice(requestData).catch(err => {
      console.error('Submission error:', err);
    });
  };

  // Сброс формы
  const handleReset = () => {
    setFormData({
      age: '',
      sex: 'male',
      bmi: '',
      children: '',
      smoker: 'no',
      // region удален
    });
    setValidationErrors({});
    setTouched({});
    resetApiState();
  };

  // Вспомогательная функция для определения ошибки поля
  const hasError = (fieldName: string): boolean => {
    return touched[fieldName] && !!validationErrors[fieldName as keyof ValidationErrors];
  };

  return (
    <div className="insurance-container">
      <div className="insurance-card">
        <h1>Калькулятор страховки</h1>
        <p className="subtitle">Заполните данные для расчета стоимости медицинской страховки</p>

        <form onSubmit={handleSubmit} noValidate>
          {/* Возраст */}
          <div className={`form-group ${hasError('age') ? 'has-error' : ''}`}>
            <label htmlFor="age">
              Возраст <span className="required">*</span>
            </label>
            <input
              type="number"
              id="age"
              name="age"
              value={formData.age}
              onChange={handleChange}
              onBlur={handleBlur}
              placeholder="18-100"
              min="18"
              max="100"
              step="1"
              disabled={loading}
            />
            {hasError('age') && (
              <div className="field-error">{validationErrors.age}</div>
            )}
          </div>

          {/* Пол */}
          <div className="form-group">
            <label>Пол <span className="required">*</span></label>
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  name="sex"
                  value="male"
                  checked={formData.sex === 'male'}
                  onChange={handleChange}
                  disabled={loading}
                />
                Мужской
              </label>
              <label>
                <input
                  type="radio"
                  name="sex"
                  value="female"
                  checked={formData.sex === 'female'}
                  onChange={handleChange}
                  disabled={loading}
                />
                Женский
              </label>
            </div>
          </div>

          {/* BMI */}
          <div className={`form-group ${hasError('bmi') ? 'has-error' : ''}`}>
            <label htmlFor="bmi">
              BMI (индекс массы тела) <span className="required">*</span>
            </label>
            <input
              type="number"
              id="bmi"
              name="bmi"
              value={formData.bmi}
              onChange={handleChange}
              onBlur={handleBlur}
              placeholder="10-50"
              step="0.1"
              min="10"
              max="50"
              disabled={loading}
            />
            {hasError('bmi') && (
              <div className="field-error">{validationErrors.bmi}</div>
            )}
            <small className="help-text">
              BMI = вес (кг) / рост² (м²)
            </small>
          </div>

          {/* Количество детей */}
          <div className={`form-group ${hasError('children') ? 'has-error' : ''}`}>
            <label htmlFor="children">
              Количество детей <span className="required">*</span>
            </label>
            <input
              type="number"
              id="children"
              name="children"
              value={formData.children}
              onChange={handleChange}
              onBlur={handleBlur}
              placeholder="0-10"
              min="0"
              max="10"
              step="1"
              disabled={loading}
            />
            {hasError('children') && (
              <div className="field-error">{validationErrors.children}</div>
            )}
          </div>

          {/* Курит или нет */}
          <div className="form-group">
            <label>Курит? <span className="required">*</span></label>
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  name="smoker"
                  value="yes"
                  checked={formData.smoker === 'yes'}
                  onChange={handleChange}
                  disabled={loading}
                />
                Да
              </label>
              <label>
                <input
                  type="radio"
                  name="smoker"
                  value="no"
                  checked={formData.smoker === 'no'}
                  onChange={handleChange}
                  disabled={loading}
                />
                Нет
              </label>
            </div>
          </div>

          {/* Общая ошибка */}
          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}

          {/* Кнопки */}
          <div className="button-group">
            <button 
              type="submit" 
              disabled={loading} 
              className="btn-primary"
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Расчет...
                </>
              ) : (
                'Рассчитать стоимость страховки'
              )}
            </button>
            <button 
              type="button" 
              onClick={handleReset} 
              className="btn-secondary"
              disabled={loading}
            >
              Сбросить
            </button>
          </div>
        </form>

        {/* Результат */}
        {price !== null && (
          <div className="result">
            <h2>Результат расчета</h2>
            <div className="price">
              <span className="price-label">Стоимость страховки:</span>
              <span className="price-value">
                ${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>
            <p className="price-note">* Ежегодная стоимость медицинской страховки</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default InsuranceForm;