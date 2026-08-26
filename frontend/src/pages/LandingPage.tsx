import React from 'react';
import { ShieldCheck, Sparkles, TrendingDown, ArrowRight, CheckCircle2, UserCheck, Eye, Layers, Lock } from 'lucide-react';

interface Props {
  onLoginClick: (role?: string) => void;
}

export const LandingPage: React.FC<Props> = ({ onLoginClick }) => {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col justify-between">
      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-24">
        
        {/* Badge */}
        <div className="flex justify-center mb-6">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-semibold backdrop-blur-md">
            <Sparkles className="w-4 h-4" />
            AI for Social Impact → AI for Education
          </div>
        </div>

        {/* Headline */}
        <div className="text-center max-w-4xl mx-auto">
          <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white leading-tight">
            Detect learning gaps <br />
            <span className="bg-gradient-to-r from-blue-400 via-indigo-400 to-emerald-400 bg-clip-text text-transparent">
              before they become learning failures.
            </span>
          </h1>
          <p className="mt-6 text-lg sm:text-xl text-slate-300 max-w-3xl mx-auto leading-relaxed">
            EduSaarthi AI continuously analyzes fragmented student learning signals, detects emerging support needs, explains evidence to educators, and generates personalized interventions — keeping teachers in the decision-making loop.
          </p>
        </div>

        {/* Demo Login Quick CTA */}
        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <button
            onClick={() => onLoginClick('STUDENT')}
            className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-xl shadow-blue-600/30 hover:scale-105 transition-all flex items-center gap-2"
          >
            <UserCheck className="w-4 h-4" />
            Demo Student View (Arjun)
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            onClick={() => onLoginClick('TEACHER')}
            className="px-6 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-sm border border-slate-700 hover:scale-105 transition-all flex items-center gap-2"
          >
            <Eye className="w-4 h-4 text-emerald-400" />
            Demo Teacher Dashboard
          </button>

          <button
            onClick={() => onLoginClick('ADMIN')}
            className="px-6 py-3 rounded-xl bg-indigo-950/80 hover:bg-indigo-900 text-indigo-300 font-semibold text-sm border border-indigo-700/50 hover:scale-105 transition-all flex items-center gap-2"
          >
            <Lock className="w-4 h-4 text-indigo-400" />
            Admin Security Hub
          </button>
        </div>

        {/* Closed Feedback Loop Diagram */}
        <div className="mt-20 glass-card p-8 border border-slate-800">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-white">The Core Closed Feedback Loop</h2>
            <p className="text-sm text-slate-400 mt-1">Continuous early detection, evidence explanation, and teacher-controlled intervention</p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-7 gap-3 text-center">
            {[
              { step: 'DETECT', desc: 'Identify score drops & repeat errors', color: 'border-red-500/50 text-red-400' },
              { step: 'EXPLAIN', desc: 'Synthesize actionable evidence', color: 'border-amber-500/50 text-amber-400' },
              { step: 'INTERVENE', desc: 'Generate 5-day roadmap', color: 'border-blue-500/50 text-blue-400' },
              { step: 'TEACHER LOOP', desc: 'Approve, Edit, or Reject', color: 'border-emerald-500/50 text-emerald-400 font-bold' },
              { step: 'LEARN', desc: 'Student completes tasks', color: 'border-indigo-500/50 text-indigo-400' },
              { step: 'REASSESS', desc: 'Targeted mastery quiz', color: 'border-purple-500/50 text-purple-400' },
              { step: 'MONITOR', desc: 'Record score recovery', color: 'border-teal-500/50 text-teal-400' },
            ].map((item, idx) => (
              <div key={idx} className={`p-4 rounded-xl bg-slate-900 border ${item.color} flex flex-col items-center justify-center`}>
                <span className="text-xs font-extrabold uppercase tracking-wider">{item.step}</span>
                <p className="text-[11px] text-slate-400 mt-2 leading-tight">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Feature Highlights Grid */}
        <div className="mt-16 grid md:grid-cols-3 gap-8">
          <div className="glass-card p-6">
            <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/30 flex items-center justify-center mb-4 text-blue-400">
              <TrendingDown className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white">Explainable Evidence Engine</h3>
            <p className="text-sm text-slate-400 mt-2 leading-relaxed">
              Never outputs vague predictions. Explains concrete signals: topic accuracy thresholds, score trajectories (e.g. 82% → 76% → 65% → 52%), and specific concept errors.
            </p>
          </div>

          <div className="glass-card p-6">
            <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center mb-4 text-emerald-400">
              <Layers className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white">Teacher-in-the-Loop Workflow</h3>
            <p className="text-sm text-slate-400 mt-2 leading-relaxed">
              AI recommendations are never imposed automatically. Teachers retain total authority to APPROVE, EDIT, or REJECT interventions before students receive them.
            </p>
          </div>

          <div className="glass-card p-6">
            <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center mb-4 text-indigo-400">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white">DevSecOps & Prompt Security</h3>
            <p className="text-sm text-slate-400 mt-2 leading-relaxed">
              Built-in RAG prompt-injection protection treats retrieved documents as DATA. Includes rate limiting, input validation, JWT auth, and active security audit logs.
            </p>
          </div>
        </div>

        {/* Responsible AI Disclaimer Banner */}
        <div className="mt-16 p-6 rounded-2xl bg-slate-900 border border-slate-800 text-center">
          <div className="flex items-center justify-center gap-2 text-slate-300 font-semibold text-sm">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            Responsible AI Commitment
          </div>
          <p className="text-xs text-slate-400 mt-2 max-w-3xl mx-auto">
            "EduSaarthi AI provides AI-assisted learning support indicators and evidence. It does NOT diagnose students, label learners as weak or incapable, or make irreversible academic decisions."
          </p>
        </div>

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-500">
        EduSaarthi AI © 2026 — AI for Social Impact. Hackathon Production MVP.
      </footer>
    </div>
  );
};
