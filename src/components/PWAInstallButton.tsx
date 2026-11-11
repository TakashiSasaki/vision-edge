import React, { useState } from 'react';
import { Download, X } from 'lucide-react';
import { usePWAInstall } from '../hooks/usePWAInstall';

export const PWAInstallButton: React.FC = () => {
  const { isInstallable, isInstalled, isIOS, install } = usePWAInstall();
  const [showIOSGuide, setShowIOSGuide] = useState(false);

  if (isInstalled) {
    return null;
  }

  if (isInstallable) {
    return (
      <button
        onClick={install}
        aria-label="アプリをインストール"
        title="PWAとして端末にインストール"
        className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-emerald-400 bg-emerald-950/60 border border-emerald-600/50 rounded-md hover:bg-emerald-900/60 hover:text-emerald-300 transition-colors whitespace-nowrap shadow-xs shadow-emerald-900/20"
      >
        <Download className="w-3.5 h-3.5 text-emerald-400" />
        <span>インストール</span>
      </button>
    );
  }

  if (isIOS) {
    return (
      <>
        <button
          onClick={() => setShowIOSGuide(true)}
          aria-label="iOSでホーム画面に追加"
          className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-slate-300 bg-slate-800/80 border border-slate-700 rounded-md hover:bg-slate-700 hover:text-white transition-colors whitespace-nowrap"
        >
          <Download className="w-3.5 h-3.5 text-blue-400" />
          <span>ホーム画面に追加</span>
        </button>

        {showIOSGuide && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
            <div className="w-full max-w-sm rounded-xl bg-slate-900 border border-slate-700 p-5 shadow-2xl text-slate-100">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <h3 className="text-sm font-semibold text-white">iPhone / iPad でのインストール</h3>
                <button
                  onClick={() => setShowIOSGuide(false)}
                  className="text-slate-400 hover:text-white p-1"
                  aria-label="案内を閉じる"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="mt-3 text-xs text-slate-300 space-y-2 leading-relaxed">
                <p>1. Safari のツールバーにある <strong>共有（四角から上矢印）</strong> ボタンをタップします。</p>
                <p>2. メニューをスクロールして <strong>「ホーム画面に追加」</strong> をタップします。</p>
                <p>3. 右上の <strong>「追加」</strong> をタップすると、ホーム画面からネイティブアプリのように起動できます。</p>
              </div>
              <button
                onClick={() => setShowIOSGuide(false)}
                className="mt-4 w-full rounded-md bg-blue-600 py-1.5 text-xs font-semibold text-white hover:bg-blue-500 transition-colors"
              >
                閉じる
              </button>
            </div>
          </div>
        )}
      </>
    );
  }

  return null;
};
