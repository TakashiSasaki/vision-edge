import type { DetectedObject } from '../types/vision';
export function confidenceDescription(object:DetectedObject):string {
  return object.confidenceKind==='semantic-softmax-mean'?'平均softmaxスコア':object.confidenceKind==='semantic-softmax-pixel'?'画素softmaxスコア':'信頼度';
}
