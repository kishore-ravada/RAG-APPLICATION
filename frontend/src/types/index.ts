export type UserRole = 'STUDENT' | 'TEACHER' | 'ADMIN';

export type IndicatorStatus = 'STRONG' | 'ON_TRACK' | 'NEEDS_ATTENTION';

export type PlanStatus = 'PROPOSED' | 'APPROVED' | 'REJECTED' | 'COMPLETED';

export type User = {
  id: number;
  email: string;
  full_name: string;
  role: UserRole;
  is_active: boolean;
};

export type AuthState = {
  user: User | null;
  token: string | null;
};

export type TrendPoint = {
  attempt_number: number;
  score: number;
  date: string;
};

export type PlanItem = {
  id: number;
  day_number: number;
  title: string;
  objective: string;
  activity_type: string;
  resource_link?: string;
  status: string;
};

export type Plan = {
  id: number;
  student_id: number;
  topic_id: number;
  topic_name: string;
  title: string;
  status: PlanStatus;
  created_by: string;
  teacher_feedback?: string;
  created_at: string;
  items: PlanItem[];
};

export type StudentDashboard = {
  student_name: string;
  grade_level: string;
  indicator_status: IndicatorStatus;
  average_score: number;
  strong_topics: string[];
  weak_topics: string[];
  recent_trend: TrendPoint[];
  active_plan?: Plan;
};

export type StudentOverviewForTeacher = {
  student_id: number;
  user_id: number;
  full_name: string;
  grade_level: string;
  average_score: number;
  indicator_status: IndicatorStatus;
  weakest_topic: string;
  evidence_snippet: string;
  recommended_action: string;
};

export type TeacherDashboard = {
  total_students: number;
  students_on_track: number;
  students_needing_attention: number;
  average_class_score: number;
  most_difficult_topics: string[];
  students: StudentOverviewForTeacher[];
};

export type LearningGap = {
  id: number;
  topic_name: string;
  indicator_status: IndicatorStatus;
  accuracy_percentage: number;
  repeated_mistakes_count: number;
  evidence_text: string;
  updated_at: string;
};

export type StudentDetail = {
  student_id: number;
  full_name: string;
  grade_level: string;
  indicator_status: IndicatorStatus;
  average_score: number;
  score_history: TrendPoint[];
  learning_gaps: LearningGap[];
  active_plans: Plan[];
};

export type SecurityEvent = {
  id: number;
  event_type: string;
  severity: string;
  description: string;
  source_ip?: string;
  timestamp: string;
};

export type AuditLog = {
  id: number;
  user_email?: string;
  action: string;
  details?: string;
  ip_address?: string;
  timestamp: string;
};

export type SecurityDashboard = {
  active_controls: Record<string, boolean>;
  total_users: number;
  failed_logins: number;
  unauthorized_attempts: number;
  security_events: SecurityEvent[];
  audit_logs: AuditLog[];
};

export type Question = {
  id: number;
  question_text: string;
  option_a: string;
  option_b: string;
  option_c: string;
  option_d: string;
  difficulty: string;
};

export type Quiz = {
  id: number;
  title: string;
  subject_id: number;
  topic_id: number;
  total_questions: number;
  questions: Question[];
};

export type QuizResult = {
  attempt_id: number;
  quiz_id: number;
  score_percentage: number;
  total_questions: number;
  correct_count: number;
  incorrect_count: number;
  indicator_status: IndicatorStatus;
  evidence_text: string;
  completed_at: string;
};

export type TutorSource = {
  title: string;
  snippet: string;
};

export type TutorResponse = {
  explanation: string;
  example: string;
  common_mistake: string;
  practice_question: string;
  sources: TutorSource[];
  is_demo_mode: boolean;
};
