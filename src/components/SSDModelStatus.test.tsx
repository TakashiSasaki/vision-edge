import {render,screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {expect,it,vi} from 'vitest';
import {ModelStatus} from './ModelStatus';
import {ModelAssetsBlockedError} from '../services/modelAssetResolver';
it('SSD asset-policy block is distinct from unrelated startup failure and retries only on deliberate action',async()=>{
  const retry=vi.fn();const {rerender}=render(<ModelStatus state={{status:'error',model:'mediapipe',variantId:'ssd-mobilenet-v2',requestId:1,stage:'task',cause:new ModelAssetsBlockedError('mediapipe','ssd-mobilenet-v2','SDK互換性確認待ち')}} onRetry={retry}/>);
  expect(screen.getByRole('alert')).toHaveTextContent('BLOCKED：');expect(retry).not.toHaveBeenCalled();await userEvent.click(screen.getByRole('button',{name:'モデルを再試行'}));expect(retry).toHaveBeenCalledTimes(1);
  rerender(<ModelStatus state={{status:'error',model:'mediapipe',variantId:'ssd-mobilenet-v2',requestId:2,stage:'wasm',cause:Error('UNRELATED_WASM_FAILURE')}} onRetry={retry}/>);expect(screen.getByRole('alert')).toHaveTextContent('モデル実行環境の読み込みに失敗');expect(screen.getByRole('alert')).not.toHaveTextContent('BLOCKED');
});
