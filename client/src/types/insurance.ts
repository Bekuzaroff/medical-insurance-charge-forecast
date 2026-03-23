// types/insurance.ts

export interface CustomerData {
  age: number | string;
  sex: 'male' | 'female';
  bmi: number | string;
  children: number | string;
  smoker: 'yes' | 'no';
  // region удален
}

export interface InsuranceRequest {
  age: number;
  sex: 'male' | 'female';
  bmi: number;
  children: number;
  smoker: 'yes' | 'no';
  // region удален
}

export interface InsuranceResponse {
  prediction: number;
}

export interface InsuranceError {
  message: string;
  status?: number;
}

export interface ValidationErrors {
  age?: string;
  bmi?: string;
  children?: string;
}