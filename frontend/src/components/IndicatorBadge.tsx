import React from 'react';
import type { IndicatorStatus } from '../types';
import { AlertTriangle, CheckCircle, TrendingUp } from 'lucide-react';

interface Props {
  status: IndicatorStatus;
  showIcon?: boolean;
}

export const IndicatorBadge: React.FC<Props> = ({ status, showIcon = true }) => {
  switch (status) {
    case 'NEEDS_ATTENTION':
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-red-500/10 text-red-400 border border-red-500/30 animate-pulse">
          {showIcon && <AlertTriangle className="w-3.5 h-3.5" />}
          Needs Attention
        </span>
      );
    case 'STRONG':
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
          {showIcon && <CheckCircle className="w-3.5 h-3.5" />}
          Strong Mastery
        </span>
      );
    case 'ON_TRACK':
    default:
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/30">
          {showIcon && <TrendingUp className="w-3.5 h-3.5" />}
          On Track
        </span>
      );
  }
};
