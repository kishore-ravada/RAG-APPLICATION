import axios from 'axios';
import type {
  StudentDashboard,
  TeacherDashboard,
  StudentDetail,
  SecurityDashboard,
  Quiz,
  QuizResult,
  TutorResponse,
  Plan
} from '../types';

const API_BASE = 'http://127.0.0.1:8080/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const authAPI = {
  login: async (email: string, password: string) => {
    const res = await api.post('/auth/login', { email, password });
    return res.data;
  },
  register: async (data: any) => {
    const res = await api.post('/auth/register', data);
    return res.data;
  },
  getMe: async () => {
    const res = await api.get('/auth/me');
    return res.data;
  },
  logout: async () => {
    await api.post('/auth/logout');
  }
};

export const studentAPI = {
  getDashboard: async (): Promise<StudentDashboard> => {
    const res = await api.get('/student/dashboard');
    return res.data;
  },
  getGaps: async () => {
    const res = await api.get('/student/learning-gaps');
    return res.data;
  },
  generatePlan: async (topicId: number = 2): Promise<Plan> => {
    const res = await api.post(`/student/learning-plan?topic_id=${topicId}`);
    return res.data;
  }
};

export const teacherAPI = {
  getDashboard: async (): Promise<TeacherDashboard> => {
    const res = await api.get('/teacher/dashboard');
    return res.data;
  },
  getStudentDetail: async (studentId: number): Promise<StudentDetail> => {
    const res = await api.get(`/teacher/students/${studentId}`);
    return res.data;
  },
  updateIntervention: async (planId: number, action: 'APPROVE' | 'REJECT' | 'EDIT', teacherFeedback?: string): Promise<Plan> => {
    const res = await api.patch(`/teacher/interventions/${planId}`, { action, teacher_feedback: teacherFeedback });
    return res.data;
  }
};

export const adminAPI = {
  getSecurityDashboard: async (): Promise<SecurityDashboard> => {
    const res = await api.get('/admin/security-dashboard');
    return res.data;
  }
};

export const tutorAPI = {
  ask: async (question: string, level: string = 'INTERMEDIATE'): Promise<TutorResponse> => {
    const res = await api.post('/tutor/ask', { question, level });
    return res.data;
  }
};

export const quizAPI = {
  generate: async (topicId: number = 2): Promise<Quiz> => {
    const res = await api.post('/quizzes/generate', { subject_id: 2, topic_id: topicId });
    return res.data;
  },
  submit: async (quizId: number, answers: Array<{ question_id: number; selected_option: string }>): Promise<QuizResult> => {
    const res = await api.post(`/quizzes/${quizId}/submit`, { answers });
    return res.data;
  }
};

export const healthAPI = {
  getHealth: async () => {
    const res = await axios.get('http://127.0.0.1:8080/health');
    return res.data;
  }
};

export default api;
