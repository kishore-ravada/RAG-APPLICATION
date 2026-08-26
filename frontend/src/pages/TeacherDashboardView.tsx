import React, { useEffect, useState } from 'react';
import { teacherAPI } from '../services/api';
import type { TeacherDashboard, StudentOverviewForTeacher, StudentDetail } from '../types';
import { IndicatorBadge } from '../components/IndicatorBadge';
import { Users, AlertTriangle, CheckCircle, TrendingUp, Eye, Check, X, Edit3, Sparkles } from 'lucide-react';
import { ResponsiveContainer, XAxis, YAxis, Tooltip, CartesianGrid, AreaChart, Area } from 'recharts';

export const TeacherDashboardView: React.FC = () => {
  const [dash, setDash] = useState<TeacherDashboard | null>(null);
  const [selectedStudentId, setSelectedStudentId] = useState<number | null>(null);
  const [studentDetail, setStudentDetail] = useState<StudentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionFeedback, setActionFeedback] = useState('');

  useEffect(() => {
    loadDashboard();
  }, []);

  const loadDashboard = () => {
    teacherAPI.getDashboard()
      .then(res => setDash(res))
      .catch(err => console.error(err))
      .finally(() => setLoading(false));
  };

  const handleOpenStudent = (id: number) => {
    setSelectedStudentId(id);
    teacherAPI.getStudentDetail(id)
      .then(res => setStudentDetail(res))
      .catch(err => console.error(err));
  };

  const handleInterventionAction = async (planId: number, action: 'APPROVE' | 'REJECT' | 'EDIT') => {
    try {
      await teacherAPI.updateIntervention(planId, action, actionFeedback || 'Reviewed and approved by Prof. Sunita Verma');
      alert(`Intervention set to ${action} successfully!`);
      if (selectedStudentId) {
        handleOpenStudent(selectedStudentId);
      }
      loadDashboard();
    } catch (err) {
      alert('Failed to update intervention status.');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  if (!dash) return null;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Teacher Intelligence Dashboard</h1>
          <p className="text-xs text-slate-400 mt-1">Computer Science & DBMS Class Performance Analytics</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-xs font-semibold">
            Teacher-in-the-Loop Active
          </span>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-6">
        <div className="glass-card p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400">Total Enrolled</span>
            <Users className="w-4 h-4 text-blue-400" />
          </div>
          <p className="text-3xl font-extrabold text-white mt-2">{dash.total_students}</p>
        </div>

        <div className="glass-card p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400">On Track</span>
            <CheckCircle className="w-4 h-4 text-emerald-400" />
          </div>
          <p className="text-3xl font-extrabold text-emerald-400 mt-2">{dash.students_on_track}</p>
        </div>

        <div className="glass-card p-5 border-l-4 border-l-red-500">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400">Needing Support</span>
            <AlertTriangle className="w-4 h-4 text-red-400" />
          </div>
          <p className="text-3xl font-extrabold text-red-400 mt-2">{dash.students_needing_attention}</p>
        </div>

        <div className="glass-card p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400">Average Class Score</span>
            <TrendingUp className="w-4 h-4 text-indigo-400" />
          </div>
          <p className="text-3xl font-extrabold text-white mt-2">{dash.average_class_score}%</p>
        </div>
      </div>

      {/* Roster & Evidence Table */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-bold text-white mb-4">Student Learning Support Roster</h2>
        
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/80 text-slate-400 uppercase font-semibold text-[10px] tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Student Name</th>
                <th className="py-3 px-4">Grade</th>
                <th className="py-3 px-4">Avg Score</th>
                <th className="py-3 px-4">Support Indicator</th>
                <th className="py-3 px-4">Evidence / Trajectory</th>
                <th className="py-3 px-4">Recommended Action</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              {dash.students.map((s) => (
                <tr key={s.student_id} className={`hover:bg-slate-800/40 transition-colors ${s.indicator_status === 'NEEDS_ATTENTION' ? 'bg-red-500/5' : ''}`}>
                  <td className="py-3.5 px-4 font-semibold text-white">{s.full_name}</td>
                  <td className="py-3.5 px-4 text-slate-400">{s.grade_level}</td>
                  <td className="py-3.5 px-4 font-bold">{s.average_score}%</td>
                  <td className="py-3.5 px-4">
                    <IndicatorBadge status={s.indicator_status} />
                  </td>
                  <td className="py-3.5 px-4 text-slate-300 max-w-xs truncate" title={s.evidence_snippet}>
                    {s.evidence_snippet}
                  </td>
                  <td className="py-3.5 px-4 text-blue-300 font-medium">{s.recommended_action}</td>
                  <td className="py-3.5 px-4 text-right">
                    <button
                      onClick={() => handleOpenStudent(s.student_id)}
                      className="px-3 py-1.5 rounded-lg bg-blue-600/20 hover:bg-blue-600 text-blue-300 hover:text-white text-xs font-semibold transition-all inline-flex items-center gap-1"
                    >
                      <Eye className="w-3.5 h-3.5" />
                      Inspect Evidence
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Student Detail & Teacher-in-the-Loop Intervention Modal */}
      {selectedStudentId && studentDetail && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="glass-card max-w-3xl w-full max-h-[90vh] overflow-y-auto p-6 border border-slate-700 shadow-2xl relative">
            
            <button
              onClick={() => { setSelectedStudentId(null); setStudentDetail(null); }}
              className="absolute top-4 right-4 p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 rounded-xl bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-400">
                <AlertTriangle className="w-6 h-6" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-xl font-bold text-white">{studentDetail.full_name}</h2>
                  <IndicatorBadge status={studentDetail.indicator_status} />
                </div>
                <p className="text-xs text-slate-400">{studentDetail.grade_level} • Target Subject: DBMS & SQL JOINs</p>
              </div>
            </div>

            {/* Score History Chart */}
            <div className="mb-6 p-4 rounded-xl bg-slate-900 border border-slate-800">
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">4-Week Assessment Trajectory</h3>
              <div className="h-40 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={studentDetail.score_history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="date" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 100]} stroke="#94a3b8" tick={{ fontSize: 11 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff' }} />
                    <Area type="monotone" dataKey="score" stroke="#ef4444" strokeWidth={2} fill="#ef4444" fillOpacity={0.2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Learning Gap Evidence Details */}
            <div className="mb-6 space-y-3">
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">Identified Evidence & Misconception Log</h3>
              {studentDetail.learning_gaps.map((g) => (
                <div key={g.id} className="p-4 rounded-xl bg-slate-900 border border-red-500/30 text-xs space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white">{g.topic_name}</span>
                    <span className="text-red-400 font-bold">Accuracy: {g.accuracy_percentage}%</span>
                  </div>
                  <p className="text-slate-300 leading-relaxed">{g.evidence_text}</p>
                </div>
              ))}
            </div>

            {/* Teacher-in-the-Loop Intervention Controls */}
            {studentDetail.active_plans.length > 0 && (
              <div className="p-5 rounded-xl bg-gradient-to-br from-slate-900 to-slate-950 border border-emerald-500/30">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-emerald-400" />
                    <h3 className="text-sm font-bold text-white">Teacher-in-the-Loop Intervention Control</h3>
                  </div>
                  <span className="text-[10px] font-extrabold uppercase px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300">
                    Status: {studentDetail.active_plans[0].status}
                  </span>
                </div>

                <p className="text-xs text-slate-300 mb-3">{studentDetail.active_plans[0].title}</p>

                <div className="mb-4">
                  <label className="block text-[11px] font-semibold text-slate-400 mb-1">Teacher Custom Note / Feedback</label>
                  <input
                    type="text"
                    value={actionFeedback}
                    onChange={(e) => setActionFeedback(e.target.value)}
                    placeholder="e.g. Focus specifically on ON clause syntax during Day 2 exercises."
                    className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-white text-xs"
                  />
                </div>

                <div className="flex items-center gap-3">
                  <button
                    onClick={() => handleInterventionAction(studentDetail.active_plans[0].id, 'APPROVE')}
                    className="flex-1 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition-all flex items-center justify-center gap-1.5"
                  >
                    <Check className="w-4 h-4" />
                    Approve Intervention
                  </button>

                  <button
                    onClick={() => handleInterventionAction(studentDetail.active_plans[0].id, 'EDIT')}
                    className="flex-1 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition-all flex items-center justify-center gap-1.5"
                  >
                    <Edit3 className="w-4 h-4" />
                    Modify & Approve
                  </button>

                  <button
                    onClick={() => handleInterventionAction(studentDetail.active_plans[0].id, 'REJECT')}
                    className="flex-1 py-2 rounded-lg bg-red-600/20 hover:bg-red-600 text-red-300 hover:text-white border border-red-500/30 font-semibold text-xs transition-all flex items-center justify-center gap-1.5"
                  >
                    <X className="w-4 h-4" />
                    Reject
                  </button>
                </div>
              </div>
            )}

          </div>
        </div>
      )}

    </div>
  );
};
