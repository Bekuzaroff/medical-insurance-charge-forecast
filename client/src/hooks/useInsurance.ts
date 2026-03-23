

import { useState, useCallback } from 'react';
import type { InsuranceRequest, InsuranceResponse, InsuranceError } from '../types/insurance';

const API_URL = 'http://localhost:8000/api/insurance-price';

export const useInsurance = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [price, setPrice] = useState<number | null>(null);

  const calculatePrice = useCallback(async (requestData: InsuranceRequest): Promise<number> => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorData: InsuranceError = await response.json();
        throw new Error(errorData.message || `Ошибка ${response.status}: ${response.statusText}`);
      }

      const data: InsuranceResponse = await response.json();
      const calculatedPrice = data.prediction ?? 0;
      setPrice(calculatedPrice);
      return calculatedPrice;
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Неизвестная ошибка';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setPrice(null);
    setError(null);
    setLoading(false);
  }, []);

  return {
    loading,
    error,
    price,
    calculatePrice,
    reset,
  };
};