import React, { useState } from 'react';
import { UserCheck, Shield, Sparkles, AlertCircle } from 'lucide-react';

interface Props {
  onLoginSuccess: (email: string, pass: string) => Promise<void>;
  defaultRole?: string;
}

export const LoginPage: React.FC<Props> = ({ onLoginSuccess, defaultRole = 'STUDENT' }) => {
  const [email, setEmail] = useState(
    defaultRole === 'TEACHER' ? 'teacher@example.com' : defaultRole === 'ADMIN' ? 'admin@example.com' : 'student@example.com'
  );
  const [password, setPassword] = useState(
    defaultRole === 'TEACHER' ? 'Teacher123!' : defaultRole === 'ADMIN' ? 'Admin123!' : 'Student123!'
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      await onLoginSuccess(email, password);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Login failed. Please check credentials.');
    } finally {
      setLoading(false);
    }
  };

  const setDemoCredentials = (role: 'STUDENT' | 'TEACHER' | 'ADMIN') => {
    if (role === 'STUDENT') {
      setEmail('student@example.com');
      setPassword('Student123!');
    } else if (role === 'TEACHER') {
      setEmail('teacher@example.com');
      setPassword('Teacher123!');
    } else {
      setEmail('admin@example.com');
      setPassword('Admin123!');
    }
  };

  return (
    <div className="min-h-[85vh] flex items-center justify-center px-4 py-12">
      <div className="max-w-md w-full glass-card p-8 border border-slate-800 shadow-2xl">
        <div className="text-center mb-6">
          <div className="w-12 h-12 rounded-xl bg-blue-600/20 border border-blue-500/30 flex items-center justify-center mx-auto mb-3 text-blue-400">
            <Sparkles className="w-6 h-6" />
          </div>
          <h2 className="text-2xl font-bold text-white">Welcome to EduSaarthi AI</h2>
          <p className="text-xs text-slate-400 mt-1">Sign in to access your AI-assisted learning portal</p>
        </div>

        {/* Quick Demo Selector */}
        <div className="mb-6 p-3 rounded-xl bg-slate-900 border border-slate-800">
          <p className="text-[11px] font-bold text-slate-300 uppercase tracking-wider text-center mb-2">Instant Demo Quick-Fill</p>
          <div className="grid grid-cols-3 gap-2">
            <button
              type="button"
              onClick={() => setDemoCredentials('STUDENT')}
              className="py-1.5 px-2 rounded-lg bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 text-[11px] font-semibold border border-blue-500/30 transition-colors"
            >
              Student
            </button>
            <button
              type="button"
              onClick={() => setDemoCredentials('TEACHER')}
              className="py-1.5 px-2 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 text-[11px] font-semibold border border-emerald-500/30 transition-colors"
            >
              Teacher
            </button>
            <button
              type="button"
              onClick={() => setDemoCredentials('ADMIN')}
              className="py-1.5 px-2 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 text-[11px] font-semibold border border-indigo-500/30 transition-colors"
            >
              Admin
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">Email Address</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-3.5 py-2 rounded-lg bg-slate-900 border border-slate-700 text-white text-sm focus:outline-none focus:border-blue-500"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full px-3.5 py-2 rounded-lg bg-slate-900 border border-slate-700 text-white text-sm focus:outline-none focus:border-blue-500"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-lg shadow-blue-600/20 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? 'Authenticating...' : 'Sign In to Portal'}
          </button>
        </form>

        <p className="text-[10px] text-center text-slate-500 mt-6">
          DEMO CREDENTIALS: student@example.com / Student123!
        </p>
      </div>
    </div>
  );
};
