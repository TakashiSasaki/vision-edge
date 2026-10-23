export interface LabelTranslation {
  ja: string;
  category: string;
}

export const COCO_LABELS_JA: Record<string, LabelTranslation> = {
  person: { ja: '人物', category: '人物' },
  bicycle: { ja: '自転車', category: '乗り物' },
  car: { ja: '自動車', category: '乗り物' },
  motorcycle: { ja: 'バイク', category: '乗り物' },
  airplane: { ja: '飛行機', category: '乗り物' },
  bus: { ja: 'バス', category: '乗り物' },
  train: { ja: '電車', category: '乗り物' },
  truck: { ja: 'トラック', category: '乗り物' },
  boat: { ja: 'ボート', category: '乗り物' },
  'traffic light': { ja: '信号機', category: '屋外' },
  'fire hydrant': { ja: '消火栓', category: '屋外' },
  'stop sign': { ja: '一時停止標識', category: '屋外' },
  'parking meter': { ja: 'パーキングメーター', category: '屋外' },
  bench: { ja: 'ベンチ', category: '家具' },
  bird: { ja: '鳥', category: '動物' },
  cat: { ja: '猫', category: '動物' },
  dog: { ja: '犬', category: '動物' },
  horse: { ja: '馬', category: '動物' },
  sheep: { ja: '羊', category: '動物' },
  cow: { ja: '牛', category: '動物' },
  elephant: { ja: 'ゾウ', category: '動物' },
  bear: { ja: 'クマ', category: '動物' },
  zebra: { ja: 'シマウマ', category: '動物' },
  giraffe: { ja: 'キリン', category: '動物' },
  backpack: { ja: 'リュックサック', category: '日用品' },
  umbrella: { ja: '傘', category: '日用品' },
  handbag: { ja: 'ハンドバッグ', category: '日用品' },
  tie: { ja: 'ネクタイ', category: '衣類' },
  suitcase: { ja: 'スーツケース', category: '日用品' },
  frisbee: { ja: 'フリスビー', category: 'スポーツ' },
  skis: { ja: 'スキー板', category: 'スポーツ' },
  snowboard: { ja: 'スノーボード', category: 'スポーツ' },
  'sports ball': { ja: 'ボール', category: 'スポーツ' },
  kite: { ja: '凧', category: 'スポーツ' },
  'baseball bat': { ja: 'バット', category: 'スポーツ' },
  'baseball glove': { ja: 'グローブ', category: 'スポーツ' },
  skateboard: { ja: 'スケートボード', category: 'スポーツ' },
  surfboard: { ja: 'サーフボード', category: 'スポーツ' },
  'tennis racket': { ja: 'テニスラケット', category: 'スポーツ' },
  bottle: { ja: 'ボトル・水筒', category: '日用品' },
  'wine glass': { ja: 'ワイングラス', category: '日用品' },
  cup: { ja: 'コップ・マグカップ', category: '日用品' },
  fork: { ja: 'フォーク', category: '日用品' },
  knife: { ja: 'ナイフ', category: '日用品' },
  spoon: { ja: 'スプーン', category: '日用品' },
  bowl: { ja: 'ボウル・器', category: '日用品' },
  banana: { ja: 'バナナ', category: '食品' },
  apple: { ja: 'リンゴ', category: '食品' },
  sandwich: { ja: 'サンドイッチ', category: '食品' },
  orange: { ja: 'オレンジ', category: '食品' },
  broccoli: { ja: 'ブロッコリー', category: '食品' },
  carrot: { ja: 'ニンジン', category: '食品' },
  'hot dog': { ja: 'ホットドッグ', category: '食品' },
  pizza: { ja: 'ピザ', category: '食品' },
  donut: { ja: 'ドーナツ', category: '食品' },
  cake: { ja: 'ケーキ', category: '食品' },
  chair: { ja: '椅子', category: '家具' },
  couch: { ja: 'ソファー', category: '家具' },
  'potted plant': { ja: '観葉植物', category: '家具' },
  bed: { ja: 'ベッド', category: '家具' },
  'dining table': { ja: 'テーブル', category: '家具' },
  toilet: { ja: 'トイレ', category: '家具' },
  tv: { ja: 'テレビ・モニター', category: '電子機器' },
  laptop: { ja: 'ノートパソコン', category: '電子機器' },
  mouse: { ja: 'マウス', category: '電子機器' },
  remote: { ja: 'リモコン', category: '電子機器' },
  keyboard: { ja: 'キーボード', category: '電子機器' },
  'cell phone': { ja: 'スマートフォン', category: '電子機器' },
  microwave: { ja: '電子レンジ', category: '家電' },
  oven: { ja: 'オーブン', category: '家電' },
  toaster: { ja: 'トースター', category: '家電' },
  sink: { ja: 'シンク・流し台', category: '家電' },
  refrigerator: { ja: '冷蔵庫', category: '家電' },
  book: { ja: '本・ノート', category: '日用品' },
  clock: { ja: '時計', category: '家具' },
  vase: { ja: '花瓶', category: '家具' },
  scissors: { ja: 'ハサミ', category: '文房具' },
  'teddy bear': { ja: 'ぬいぐるみ', category: '玩具' },
  'hair drier': { ja: 'ドライヤー', category: '家電' },
  toothbrush: { ja: '歯ブラシ', category: '日用品' },
};

export const ADE20K_LABELS_JA: Record<string, LabelTranslation> = {
  wall: { ja: '壁面', category: '背景構造' },
  building: { ja: '建物', category: '屋外' },
  sky: { ja: '空', category: '屋外' },
  floor: { ja: '床', category: '背景構造' },
  tree: { ja: '樹木', category: '自然' },
  ceiling: { ja: '天井', category: '背景構造' },
  road: { ja: '道路', category: '屋外' },
  bed: { ja: 'ベッド', category: '家具' },
  windowpane: { ja: '窓ガラス', category: '背景構造' },
  cabinet: { ja: '戸棚・キャビネット', category: '家具' },
  sidewalk: { ja: '歩道', category: '屋外' },
  person: { ja: '人物', category: '人物' },
  door: { ja: 'ドア', category: '背景構造' },
  table: { ja: '机・テーブル', category: '家具' },
  plant: { ja: '植物', category: '自然' },
  curtain: { ja: 'カーテン', category: 'インテリア' },
  chair: { ja: '椅子', category: '家具' },
  car: { ja: '自動車', category: '乗り物' },
  sofa: { ja: 'ソファー', category: '家具' },
  shelf: { ja: '本棚・棚', category: '家具' },
  mirror: { ja: '鏡', category: 'インテリア' },
  rug: { ja: 'ラグ・絨毯', category: 'インテリア' },
  desk: { ja: 'デスク', category: '家具' },
  lamp: { ja: '照明器具', category: '家具' },
  screen: { ja: '画面・ディスプレイ', category: '電子機器' },
  computer: { ja: 'コンピュータ', category: '電子機器' },
  monitor: { ja: 'モニター', category: '電子機器' },
  glass: { ja: 'コップ・グラス', category: '日用品' }
};

export function getLabelInfo(label: string, isADE20K = false): LabelTranslation {
  const normalized = label.toLowerCase().trim();

  if (isADE20K) {
    if (ADE20K_LABELS_JA[normalized]) return ADE20K_LABELS_JA[normalized];
    for (const [key, val] of Object.entries(ADE20K_LABELS_JA)) {
      if (normalized.includes(key)) return val;
    }
  }

  if (COCO_LABELS_JA[normalized]) return COCO_LABELS_JA[normalized];

  for (const [key, val] of Object.entries(COCO_LABELS_JA)) {
    if (normalized.includes(key)) return val;
  }

  const fallbackJa = normalized.charAt(0).toUpperCase() + normalized.slice(1);
  return {
    ja: fallbackJa,
    category: 'その他'
  };
}

const COLOR_PALETTE = [
  '#10B981', '#3B82F6', '#8B5CF6', '#F59E0B', 
  '#EC4899', '#06B6D4', '#F97316', '#14B8A6', 
  '#6366F1', '#84CC16', '#E11D48', '#0EA5E9'
];

export function getCategoryColor(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = label.charCodeAt(i) + ((hash << 5) - hash);
  }
  const index = Math.abs(hash) % COLOR_PALETTE.length;
  return COLOR_PALETTE[index];
}
