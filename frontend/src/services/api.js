const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

const getHeaders = () => {
  const token = localStorage.getItem('token');
  return {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': token } : {}),
  };
};

export const api = {
  // Auth endpoints
  login: async (username, password) => {
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Login failed');
    }
    return response.json();
  },

  getCurrentUser: async () => {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
      method: 'GET',
      headers: getHeaders(),
    });
    if (!response.ok) {
      throw new Error('Failed to get current user');
    }
    return response.json();
  },

  getPatientPortalOverview: async () => {
    const response = await fetch(`${API_BASE_URL}/patient-portal/overview`, {
      method: 'GET',
      headers: getHeaders(),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Không thể tải hồ sơ bệnh nhân.');
    }
    return response.json();
  },

  // Patients endpoints
  getPatients: async (searchQuery = '') => {
    const url = searchQuery
      ? `${API_BASE_URL}/patients/?search=${encodeURIComponent(searchQuery)}`
      : `${API_BASE_URL}/patients/`;
    const response = await fetch(url, {
      method: 'GET',
      headers: getHeaders(),
    });
    if (!response.ok) {
      throw new Error('Failed to fetch patients');
    }
    return response.json();
  },

  createPatient: async (patientData) => {
    const response = await fetch(`${API_BASE_URL}/patients/`, {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify(patientData),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to create patient');
    }
    return response.json();
  },

  getPatientById: async (patientId) => {
    const response = await fetch(`${API_BASE_URL}/patients/${patientId}`, {
      method: 'GET',
      headers: getHeaders(),
    });
    if (!response.ok) {
      throw new Error('Patient not found');
    }
    return response.json();
  },

  // Screening history for a patient
  getScreeningModels: async () => {
    const response = await fetch(`${API_BASE_URL}/screenings/models`, {
      method: 'GET',
      headers: getHeaders(),
    });
    if (!response.ok) {
      throw new Error('Không thể tải danh sách model nhận diện.');
    }
    return response.json();
  },

  getPatientScreenings: async (patientId) => {
    const response = await fetch(`${API_BASE_URL}/screenings/patient/${patientId}`, {
      method: 'GET',
      headers: getHeaders(),
    });
    if (!response.ok) throw new Error('Failed to fetch screening history');
    return response.json();
  },

  getScreeningDetail: async (screeningId) => {
    const response = await fetch(`${API_BASE_URL}/screenings/${screeningId}`, {
      method: 'GET',
      headers: getHeaders(),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Không thể tải chi tiết lần khám.');
    }
    return response.json();
  },

  reviewScreening: async (screeningId, reviewData) => {
    const response = await fetch(`${API_BASE_URL}/reviews/${screeningId}`, {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify(reviewData),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Không thể lưu kết luận bác sĩ.');
    }
    return response.json();
  },

  // Recall / follow-up appointments for a patient
  getPatientRecalls: async (patientId) => {
    const response = await fetch(`${API_BASE_URL}/patients/${patientId}/recalls`, {
      method: 'GET',
      headers: getHeaders(),
    });
    if (!response.ok) throw new Error('Failed to fetch recall schedule');
    return response.json();
  },

  uploadScreening: async (patientId, files, gradingModel = 'grading') => {
    const form = new FormData();
    form.append('patient_id', patientId);
    form.append('grading_model', gradingModel);
    Object.entries(files).forEach(([key, file]) => form.append(key, file));
    const token = localStorage.getItem('token');
    const response = await fetch(`${API_BASE_URL}/screenings/upload`, {
      method: 'POST',
      headers: token ? { Authorization: token } : {},
      body: form,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      const detail = typeof error.detail === 'string' ? error.detail : error.detail?.message;
      throw new Error(detail || 'Không thể phân tích bộ ảnh sàng lọc.');
    }
    return response.json();
  },
  createQuickRecall: async (patientId, recallData) => {
    const response = await fetch(`${API_BASE_URL}/patients/${patientId}/recalls`, {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify(recallData),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Không thể tạo lịch tái khám.');
    }
    return response.json();
  },
};
