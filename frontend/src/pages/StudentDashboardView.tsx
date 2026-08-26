import React, { useEffect, useState } from 'react';
import { studentAPI } from '../services/api';
import type { StudentDashboard, Plan } from '../types';
import { IndicatorBadge } from '../components/IndicatorBadge';
import { BookOpen, Sparkles, TrendingDown, Clock, ArrowRight } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

interface Props {
  onNavigate: (view: string) => void;
}

export const StudentDashboardView: React.FC<Props> = ({ onNavigate }) => {
  const [data, setData] = useState<StudentDashboard | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    studentAPI.getDashboard()
      .then(res => setData(res))
      .catch(err => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Welcome & Status Banner */}
      <div className="glass-card p-6 border-l-4 border-l-red-500 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white">Welcome back, {data.student_name}</h1>
            <IndicatorBadge status={data.indicator_status} />
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Learning Support Status: <span className="font-semibold text-slate-200">Needs Attention</span> — Early topic-level practice is recommended.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate('tutor')}
            className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-lg shadow-blue-600/20 transition-all flex items-center gap-1.5"
          >
            <BookOpen className="w-4 h-4" />
            Ask AI Tutor
          </button>
          <button
            onClick={() => onNavigate('quiz')}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-all flex items-center gap-1.5"
          >
            <Sparkles className="w-4 h-4 text-blue-400" />
            Take Practice Quiz
          </button>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        <div className="glass-card p-5">
          <p className="text-xs font-semibold text-slate-400">Average Quiz Score</p>
          <div className="flex items-baseline gap-2 mt-2">
            <span className="text-3xl font-extrabold text-white">{data.average_score}%</span>
            <span className="text-xs text-red-400 font-semibold flex items-center">
              <TrendingDown className="w-3.5 h-3.5 mr-0.5" /> -30% trajectory
            </span>
          </div>
        </div>

        <div className="glass-card p-5">
          <p className="text-xs font-semibold text-slate-400">Strong Mastery Topics</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {data.strong_topics.map((t, idx) => (
              <span key={idx} className="px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-300 text-xs font-medium border border-emerald-500/20">
                {t}
              </span>
            ))}
          </div>
        </div>

        <div className="glass-card p-5">
          <p className="text-xs font-semibold text-slate-400">Focus Support Topics</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {data.weak_topics.map((t, idx) => (
              <span key={idx} className="px-2.5 py-1 rounded-md bg-red-500/10 text-red-300 text-xs font-medium border border-red-500/20">
                {t}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Performance Trend Chart */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-bold text-white">Assessment Performance Trajectory</h2>
            <p className="text-xs text-slate-400">Detected trend: Consecutive score decline across 4 recent assessments</p>
          </div>
          <span className="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-medium">
            DBMS & SQL JOINs
          </span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data.recent_trend}>
              <defs>
                <linearGradient id="scoreColor" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="date" stroke="#94a3b8" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} stroke="#94a3b8" tick={{ fontSize: 12 }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff' }}
                formatter={(val: any) => [`${val}%`, 'Score']}
              />
              <Area type="monotone" dataKey="score" stroke="#ef4444" strokeWidth={3} fillOpacity={1} fill="url(#scoreColor)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Active 5-Day Learning Support Intervention */}
      {data.active_plan && (
        <div className="glass-card p-6 border border-blue-500/30">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-white">{data.active_plan.title}</h2>
                <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${
                  data.active_plan.status === 'APPROVED' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                }`}>
                  {data.active_plan.status === 'APPROVED' ? 'Teacher Approved' : 'Teacher Review Pending'}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">Structured 5-day practice roadmap designed to resolve SQL JOIN misconceptions.</p>
            </div>
            
            <button
              onClick={() => onNavigate('quiz')}
              className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold transition-all flex items-center gap-1.5 self-start md:self-auto"
            >
              Start Reassessment Quiz
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {data.active_plan.items.map((item) => (
              <div key={item.id} className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[11px] font-bold uppercase text-blue-400">Day {item.day_number}</span>
                    <Clock className="w-3.5 h-3.5 text-slate-500" />
                  </div>
                  <h3 className="text-xs font-bold text-white line-clamp-1">{item.title}</h3>
                  <p className="text-[11px] text-slate-400 mt-1 leading-snug">{item.objective}</p>
                </div>
                
                <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between">
                  <span className="text-[10px] text-slate-500 font-medium">{item.activity_type}</span>
                  <button 
                    onClick={() => onNavigate('tutor')} 
                    className="text-[11px] text-blue-400 hover:underline font-semibold"
                  >
                    Open Task
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
};
