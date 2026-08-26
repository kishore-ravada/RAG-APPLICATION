import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { LandingPage } from './pages/LandingPage';
import { LoginPage } from './pages/LoginPage';
import { StudentDashboardView } from './pages/StudentDashboardView';
import { TeacherDashboardView } from './pages/TeacherDashboardView';
import { AdminDashboardView } from './pages/AdminDashboardView';
import { AITutorView } from './pages/AITutorView';
import { QuizView } from './pages/QuizView';
import { authAPI, healthAPI } from './services/api';
import type { User } from './types';

export const App: React.FC = () => {
  const [user, setUser] = useState<User | null>(null);
  const [currentView, setCurrentView] = useState<string>('landing');
  const [loginDefaultRole, setLoginDefaultRole] = useState<string>('STUDENT');
  // Demo mode is true until the health check confirms the AI is configured
  const [isDemoMode, setIsDemoMode] = useState<boolean>(true);

  // Restore session from localStorage on mount and check AI status
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      authAPI.getMe()
        .then((u) => {
          setUser(u);
          setCurrentView('dashboard');
        })
        .catch(() => {
          localStorage.removeItem('token');
          setUser(null);
        });
    }

    // Reflect actual AI configuration status in the demo mode banner
    healthAPI.getHealth()
      .then((data) => {
        setIsDemoMode(data.ai !== 'configured');
      })
      .catch(() => {
        // If health check fails, assume demo mode (safe default)
        setIsDemoMode(true);
      });
  }, []);

  const handleLoginSuccess = async (email: string, pass: string) => {
    const data = await authAPI.login(email, pass);
    localStorage.setItem('token', data.access_token);
    setUser({
      id: data.user_id,
      email: data.email,
      full_name: data.full_name,
      role: data.role,
      is_active: true
    });
    setCurrentView('dashboard');
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setUser(null);
    setCurrentView('landing');
  };

  const handleQuickLoginClick = (role: string = 'STUDENT') => {
    setLoginDefaultRole(role);
    setCurrentView('login');
  };

  const renderCurrentView = () => {
    switch (currentView) {
      case 'landing':
        return <LandingPage onLoginClick={handleQuickLoginClick} />;
      case 'login':
        return <LoginPage onLoginSuccess={handleLoginSuccess} defaultRole={loginDefaultRole} />;
      case 'tutor':
        return <AITutorView />;
      case 'quiz':
        return <QuizView />;
      case 'admin':
        return user?.role === 'ADMIN'
          ? <AdminDashboardView />
          : <LandingPage onLoginClick={handleQuickLoginClick} />;
      case 'dashboard':
      default:
        if (!user) return <LandingPage onLoginClick={handleQuickLoginClick} />;
        if (user.role === 'TEACHER') return <TeacherDashboardView />;
        if (user.role === 'ADMIN') return <AdminDashboardView />;
        return <StudentDashboardView onNavigate={(v) => setCurrentView(v)} />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar
        user={user}
        currentView={currentView}
        onNavigate={(v) => setCurrentView(v)}
        onLogout={handleLogout}
        isDemoMode={isDemoMode}
      />

      <main className="flex-grow">
        {renderCurrentView()}
      </main>
    </div>
  );
};

export default App;
