import React, { useState } from 'react';
import { quizAPI } from '../services/api';
import type { Quiz, QuizResult } from '../types';
import { IndicatorBadge } from '../components/IndicatorBadge';
import { Sparkles, CheckCircle2, ArrowRight, RefreshCw, Award } from 'lucide-react';

export const QuizView: React.FC = () => {
  const [quiz, setQuiz] = useState<Quiz | null>(null);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [result, setResult] = useState<QuizResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const handleGenerateQuiz = async (topicId: number = 2) => {
    setLoading(true);
    setResult(null);
    setAnswers({});
    try {
      const q = await quizAPI.generate(topicId);
      setQuiz(q);
    } catch (err) {
      alert('Failed to generate quiz.');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectOption = (questionId: number, option: string) => {
    setAnswers(prev => ({ ...prev, [questionId]: option }));
  };

  const handleSubmitQuiz = async () => {
    if (!quiz) return;
    if (Object.keys(answers).length < quiz.questions.length) {
      alert('Please answer all questions before submitting.');
      return;
    }

    setSubmitting(true);
    try {
      const payload = Object.entries(answers).map(([qId, opt]) => ({
        question_id: Number(qId),
        selected_option: opt
      }));
      const res = await quizAPI.submit(quiz.id, payload);
      setResult(res);
    } catch (err) {
      alert('Failed to submit quiz.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Targeted Concept Assessment</h1>
          <p className="text-xs text-slate-400 mt-1">Evaluates accuracy & calculates learning-support indicators deterministically</p>
        </div>

        {!quiz && (
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleGenerateQuiz(2)}
              className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition-all flex items-center gap-1.5"
            >
              <Sparkles className="w-4 h-4" />
              Generate DBMS SQL JOIN Quiz
            </button>
          </div>
        )}
      </div>

      {/* Loading state */}
      {loading && (
        <div className="flex items-center justify-center min-h-[40vh]">
          <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      )}

      {/* Quiz Start Screen */}
      {!quiz && !loading && !result && (
        <div className="glass-card p-12 text-center border border-slate-800">
          <div className="w-16 h-16 rounded-2xl bg-blue-600/20 border border-blue-500/30 flex items-center justify-center mx-auto mb-4 text-blue-400">
            <Award className="w-8 h-8" />
          </div>
          <h2 className="text-xl font-bold text-white">Ready for your Diagnostic Assessment?</h2>
          <p className="text-xs text-slate-400 max-w-md mx-auto mt-2 leading-relaxed">
            Test your understanding of SQL JOIN syntax and concept boundaries. Scores are calculated deterministically by backend services.
          </p>

          <div className="mt-6 flex justify-center gap-3">
            <button
              onClick={() => handleGenerateQuiz(2)}
              className="px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-xl shadow-blue-600/20 transition-all flex items-center gap-2"
            >
              Start 5-Question SQL JOIN Quiz
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Quiz Questions View */}
      {quiz && !result && (
        <div className="space-y-6">
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between">
            <span className="font-bold text-white text-sm">{quiz.title}</span>
            <span className="text-xs text-slate-400 font-semibold">
              {Object.keys(answers).length} / {quiz.questions.length} Answered
            </span>
          </div>

          {quiz.questions.map((q, idx) => (
            <div key={q.id} className="glass-card p-6 border border-slate-800">
              <div className="flex items-start justify-between gap-4 mb-4">
                <span className="text-xs font-bold text-blue-400 uppercase tracking-wider">Question {idx + 1}</span>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                  {q.difficulty}
                </span>
              </div>

              <p className="text-sm font-semibold text-white mb-4 leading-relaxed">{q.question_text}</p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {[
                  { key: 'A', text: q.option_a },
                  { key: 'B', text: q.option_b },
                  { key: 'C', text: q.option_c },
                  { key: 'D', text: q.option_d },
                ].map((opt) => {
                  const isSelected = answers[q.id] === opt.key;
                  return (
                    <button
                      key={opt.key}
                      onClick={() => handleSelectOption(q.id, opt.key)}
                      className={`p-3 rounded-xl text-left text-xs font-medium transition-all flex items-center gap-3 border ${
                        isSelected 
                          ? 'bg-blue-600/20 border-blue-500 text-white font-bold' 
                          : 'bg-slate-900/60 border-slate-800 text-slate-300 hover:bg-slate-800'
                      }`}
                    >
                      <span className={`w-6 h-6 rounded-lg flex items-center justify-center font-bold text-[11px] ${
                        isSelected ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400'
                      }`}>
                        {opt.key}
                      </span>
                      <span>{opt.text}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}

          <div className="flex justify-end gap-3 pt-4">
            <button
              onClick={handleSubmitQuiz}
              disabled={submitting}
              className="px-8 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-xl shadow-blue-600/20 transition-all flex items-center gap-2 disabled:opacity-50"
            >
              {submitting ? 'Evaluating...' : 'Submit Answers'}
              <CheckCircle2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Quiz Result View */}
      {result && (
        <div className="glass-card p-8 border border-slate-800 text-center space-y-6">
          
          <div className="flex justify-center">
            <IndicatorBadge status={result.indicator_status} />
          </div>

          <div>
            <span className="text-5xl font-extrabold text-white">{result.score_percentage}%</span>
            <p className="text-xs text-slate-400 mt-2">
              Score calculated deterministically: {result.correct_count} Correct / {result.incorrect_count} Incorrect
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-left max-w-lg mx-auto text-xs space-y-2">
            <span className="font-bold text-slate-200 block uppercase tracking-wider text-[11px]">Actionable Evidence</span>
            <p className="text-slate-300 leading-relaxed">{result.evidence_text}</p>
          </div>

          <div className="flex justify-center gap-4 pt-4">
            <button
              onClick={() => handleGenerateQuiz(2)}
              className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-lg shadow-blue-600/20 transition-all flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              Retake Reassessment Quiz
            </button>
          </div>

        </div>
      )}

    </div>
  );
};
