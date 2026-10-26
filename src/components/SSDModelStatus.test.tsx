import {render,screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {expect,it,vi} from 'vitest';
import {ModelStatus} from './ModelStatus';
import {ModelComparisonTable} from './ModelComparisonTable';
import {ModelAssetsBlockedError} from '../services/modelAssetResolver';
it('SSD asset-policy block is distinct from unrelated startup failure and retries only on deliberate action',async()=>{
  const retry=vi.fn();const {rerender}=render(<ModelStatus state={{status:'error',model:'mediapipe',variantId:'ssd-mobilenet-v2',requestId:1,stage:'task',cause:new ModelAssetsBlockedError('mediapipe','ssd-mobilenet-v2','SDK互換性確認待ち')}} onRetry={retry}/>);
  expect(screen.getByRole('alert')).toHaveTextContent('BLOCKED：');expect(retry).not.toHaveBeenCalled();await userEvent.click(screen.getByRole('button',{name:'モデルを再試行'}));expect(retry).toHaveBeenCalledTimes(1);
  rerender(<ModelStatus state={{status:'error',model:'mediapipe',variantId:'ssd-mobilenet-v2',requestId:2,stage:'wasm',cause:Error('UNRELATED_WASM_FAILURE')}} onRetry={retry}/>);expect(screen.getByRole('alert')).toHaveTextContent('モデル実行環境の読み込みに失敗');expect(screen.getByRole('alert')).not.toHaveTextContent('BLOCKED');
});

it('SSD specification uses the actual remaining SDK prerequisite and selects the same existing detector variant',async()=>{
  const select=vi.fn(),close=vi.fn();render(<ModelComparisonTable isOpen={true} onClose={close} activeModel="mediapipe" activeVariant="lite0" onSelectModelAndVariant={select}/>);
  expect(screen.getByRole('dialog')).toHaveTextContent('SDK0.10.14での実行互換性が未検証');expect(screen.getByRole('dialog')).not.toHaveTextContent('実体未確認');
  await userEvent.click(screen.getByRole('button',{name:/SSD MobileNetV2/}));expect(select).toHaveBeenCalledWith('mediapipe','ssd-mobilenet-v2');expect(close).toHaveBeenCalledTimes(1);
});
