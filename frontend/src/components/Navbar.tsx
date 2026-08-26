import React from 'react';
import type { User } from '../types';
import { Shield, Sparkles, LogOut, BookOpen, UserCheck, Activity } from 'lucide-react';

interface Props {
  user: User | null;
  currentView: string;
  onNavigate: (view: string) => void;
  onLogout: () => void;
  isDemoMode: boolean;
}

export const Navbar: React.FC<Props> = ({ user, currentView, onNavigate, onLogout, isDemoMode }) => {
  return (
    <header className="bg-slate-900/90 border-b border-slate-800 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand */}
        <div 
          onClick={() => onNavigate(user ? 'dashboard' : 'landing')}
          className="flex items-center gap-3 cursor-pointer group"
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20 group-hover:scale-105 transition-transform">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg text-white tracking-tight">EduSaarthi <span className="text-blue-400">AI</span></span>
              {isDemoMode && (
                <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                  DEMO AI MODE
                </span>
              )}
            </div>
            <p className="text-[10px] text-slate-400 font-medium">Early Learning-Gap Detection</p>
          </div>
        </div>

        {/* Navigation Links */}
        {user ? (
          <nav className="flex items-center gap-1 sm:gap-2">
            <button
              onClick={() => onNavigate('dashboard')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                currentView === 'dashboard' ? 'bg-blue-600 text-white' : 'text-slate-300 hover:bg-slate-800'
              }`}
            >
              <Activity className="w-4 h-4" />
              Dashboard
            </button>

            {user.role === 'STUDENT' && (
              <>
                <button
                  onClick={() => onNavigate('tutor')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                    currentView === 'tutor' ? 'bg-blue-600 text-white' : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  <BookOpen className="w-4 h-4" />
                  AI Tutor
                </button>
                <button
                  onClick={() => onNavigate('quiz')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                    currentView === 'quiz' ? 'bg-blue-600 text-white' : 'text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  <Sparkles className="w-4 h-4" />
                  Take Quiz
                </button>
              </>
            )}

            {user.role === 'ADMIN' && (
              <button
                onClick={() => onNavigate('admin')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                  currentView === 'admin' ? 'bg-indigo-600 text-white' : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                <Shield className="w-4 h-4" />
                Security Dashboard
              </button>
            )}

            {/* Profile Info & Logout */}
            <div className="h-4 w-[1px] bg-slate-800 mx-2 hidden sm:block" />
            <div className="flex items-center gap-3">
              <div className="text-right hidden sm:block">
                <p className="text-xs font-semibold text-slate-200">{user.full_name}</p>
                <span className="text-[10px] text-blue-400 font-bold uppercase">{user.role}</span>
              </div>
              <button
                onClick={onLogout}
                title="Logout"
                className="p-2 rounded-lg text-slate-400 hover:text-red-400 hover:bg-slate-800 transition-colors"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          </nav>
        ) : (
          <div className="flex items-center gap-3">
            <button
              onClick={() => onNavigate('login')}
              className="px-4 py-2 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/20 transition-all flex items-center gap-1.5"
            >
              <UserCheck className="w-4 h-4" />
              Demo Access / Login
            </button>
          </div>
        )}

      </div>
    </header>
  );
};
