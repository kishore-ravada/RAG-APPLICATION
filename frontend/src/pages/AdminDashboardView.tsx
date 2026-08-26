import React, { useEffect, useState } from 'react';
import { adminAPI, healthAPI } from '../services/api';
import type { SecurityDashboard } from '../types';
import { ShieldCheck, Activity, AlertOctagon, ListFilter, CheckCircle2 } from 'lucide-react';

export const AdminDashboardView: React.FC = () => {
  const [data, setData] = useState<SecurityDashboard | null>(null);
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      adminAPI.getSecurityDashboard(),
      healthAPI.getHealth()
    ])
      .then(([secRes, healthRes]) => {
        setData(secRes);
        setHealth(healthRes);
      })
      .catch(err => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-indigo-500"></div>
      </div>
    );
  }

  if (!data) return null;

  const controlLabels: Record<string, string> = {
    authentication: 'JWT Authentication & Password Hashing',
    rbac: 'Role-Based Access Control (RBAC)',
    input_validation: 'Input Validation & Sanitization',
    rate_limiting: 'Sliding Window Rate Limiting',
    prompt_injection_protection: 'RAG Prompt Injection Guard',
    audit_logging: 'Audit Trail Logging',
    secret_management: 'Environment Secret Isolation',
    ai_fallback_demo_mode: 'AI Fallback & Demo Mode',
    health_monitoring: 'System Health Check Endpoint'
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold text-white">Admin Security & Infrastructure Dashboard</h1>
            <span className="px-2.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 text-xs font-bold uppercase">
              DevSecOps Verified
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">Real-time threat monitoring, audit logs, and security control status</p>
        </div>

        {health && (
          <div className="flex items-center gap-3 glass-card px-4 py-2 text-xs">
            <Activity className="w-4 h-4 text-emerald-400" />
            <div>
              <span className="font-bold text-white">System Health: {health.status}</span>
              <p className="text-[10px] text-slate-400">DB: {health.database} | AI: {health.ai}</p>
            </div>
          </div>
        )}
      </div>

      {/* Active Security Controls Checklist */}
      <div className="glass-card p-6 border-l-4 border-l-indigo-500">
        <h2 className="text-lg font-bold text-white mb-1 flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-indigo-400" />
          Active DevSecOps Security Controls
        </h2>
        <p className="text-xs text-slate-400 mb-6">Strictly displays status for active runtime controls.</p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(data.active_controls).map(([key, isEnabled]) => (
            <div key={key} className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between">
              <span className="text-xs font-medium text-slate-200">{controlLabels[key] || key}</span>
              {isEnabled ? (
                <span className="inline-flex items-center gap-1 text-[11px] font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 rounded">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Active ✓
                </span>
              ) : (
                <span className="text-[11px] font-bold text-slate-500">Disabled</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        <div className="glass-card p-5">
          <p className="text-xs font-semibold text-slate-400">Registered Users</p>
          <p className="text-3xl font-extrabold text-white mt-2">{data.total_users}</p>
        </div>

        <div className="glass-card p-5">
          <p className="text-xs font-semibold text-slate-400">Failed Login Attempts</p>
          <p className="text-3xl font-extrabold text-amber-400 mt-2">{data.failed_logins}</p>
        </div>

        <div className="glass-card p-5">
          <p className="text-xs font-semibold text-slate-400">Security Threats Intercepted</p>
          <p className="text-3xl font-extrabold text-indigo-400 mt-2">{data.unauthorized_attempts}</p>
        </div>
      </div>

      {/* Live Logs & Events Grid */}
      <div className="grid md:grid-cols-2 gap-8">
        
        {/* Security Threat Log */}
        <div className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <AlertOctagon className="w-5 h-5 text-red-400" />
            <h2 className="text-base font-bold text-white">Security Events & Threat Log</h2>
          </div>

          <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
            {data.security_events.map((evt) => (
              <div key={evt.id} className="p-3 rounded-xl bg-slate-900 border border-red-500/30 text-xs">
                <div className="flex items-center justify-between text-red-400 font-bold mb-1">
                  <span>{evt.event_type}</span>
                  <span className="text-[10px] text-slate-500">{new Date(evt.timestamp).toLocaleTimeString()}</span>
                </div>
                <p className="text-slate-300 leading-snug">{evt.description}</p>
                <span className="text-[10px] text-slate-500 mt-1 block">Source IP: {evt.source_ip || '127.0.0.1'}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Audit Activity Log */}
        <div className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <ListFilter className="w-5 h-5 text-blue-400" />
            <h2 className="text-base font-bold text-white">Real-Time Audit Trail</h2>
          </div>

          <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
            {data.audit_logs.map((log) => (
              <div key={log.id} className="p-3 rounded-xl bg-slate-900 border border-slate-800 text-xs">
                <div className="flex items-center justify-between font-bold text-slate-200 mb-1">
                  <span className="text-blue-400">{log.action}</span>
                  <span className="text-[10px] text-slate-500">{new Date(log.timestamp).toLocaleTimeString()}</span>
                </div>
                <p className="text-slate-400">{log.details}</p>
                <div className="flex items-center justify-between text-[10px] text-slate-500 mt-1">
                  <span>User: {log.user_email || 'System'}</span>
                  <span>IP: {log.ip_address || '127.0.0.1'}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>

    </div>
  );
};
