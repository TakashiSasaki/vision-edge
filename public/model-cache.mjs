const integrityLabels = { pinned: '固定ハッシュ一致', 'self-recorded': '取得時ハッシュ一致', invalid: '検証失敗', 'not-present': '未保存', pending: '検証待ち', unknown: '不明' };
const stateLabels = { cached: '配信準備済み', downloading: '取得中', unavailable: '未準備' };
const families = { 'selfie-segmenter': 'Selfie Segmenter', mediapipe: 'EfficientDet', 'yolo-seg': 'YOLO', segformer: 'SegFormer', 'image-classifier': 'EfficientNet（画像分類）' };
const safeName = value => typeof value === 'string' && /^[a-zA-Z0-9_-]{1,80}$/.test(value);
const safeFile = value => typeof value === 'string' && /^[a-zA-Z0-9_-][a-zA-Z0-9._-]{0,159}$/.test(value) && !value.includes('..');
// A verification success belongs to a cached, present, verified byte record.
// Non-cached states have no verified size/hash; only their state-specific
// pending/absent/failed/unknown observations may be displayed as coherent.
function coherentState(value, cached) {
  if (cached) return true;
  if (value.status === 'cached' || value.size !== null || value.sha256 !== null || (value.error !== null && (typeof value.error !== 'string' || !value.error.length))) return false;
  if (value.status === 'downloading') return value.integrity === 'pending' && value.filePresent === null && value.error === null;
  if (value.status !== 'unavailable') return false;
  if (value.integrity === 'not-present') return value.filePresent === false;
  if (value.integrity === 'invalid') return value.filePresent !== null && value.error !== null;
  return value.integrity === 'unknown' && value.filePresent === null;
}
export function inventorySnapshot(payload) {
  if (!payload || payload.enabled !== true || payload.schemaVersion !== 2 || !Number.isInteger(payload.catalogCount) || payload.catalogCount < 0 || !payload.variants || typeof payload.variants !== 'object' || Array.isArray(payload.variants) || Object.keys(payload.variants).length !== payload.catalogCount || typeof payload.observedAt !== 'string' || !Number.isFinite(Date.parse(payload.observedAt))) throw new Error('Unavailable status API');
  let invalid = 0;
  const rows = Object.entries(payload.variants).map(([key, value]) => {
    const identity = value && Object.hasOwn(families, value.id) && safeName(value.variantId) && key === `${value.id}:${value.variantId}` && safeFile(value.file);
    const shape = identity && Object.hasOwn(stateLabels, value.status) && Object.hasOwn(integrityLabels, value.integrity) && [true, false, null].includes(value.filePresent);
    const cached = shape && value.status === 'cached' && value.filePresent === true && ['pinned', 'self-recorded'].includes(value.integrity) && Number.isSafeInteger(value.size) && value.size > 0 && typeof value.sha256 === 'string' && /^[a-f0-9]{64}$/.test(value.sha256) && value.error === null && (value.integrity !== 'pinned' || (value.sha256 === value.expectedSha256 && value.size === value.expectedSize));
    const valid = shape && coherentState(value, cached);
    if (!valid) invalid++;
    return { family: identity ? families[value.id] : '不明なモデル', variant: identity ? value.variantId : '不明', file: identity ? value.file : '不正なファイル名（非表示）', size: cached ? `${value.size.toLocaleString('ja-JP')} bytes` : '—', presence: shape ? (value.filePresent === true ? 'あり' : value.filePresent === false ? 'なし' : '不明') : '不明', readiness: valid ? stateLabels[value.status] : '不明（不正な状態）', integrity: valid ? integrityLabels[value.integrity] : '不明', error: value?.error ? '取得・検証エラーあり（詳細は管理ログ）' : '—', cached };
  });
  return { rows, invalid, observedAt: payload.observedAt };
}
function table(document, host, rows, empty) {
  host.replaceChildren();
  if (!rows.length) { const p = document.createElement('p'); p.textContent = empty; host.append(p); return; }
  const headers = ['モデル', 'バリアント', 'ファイル名', '検証済みサイズ', '実ファイル', '準備状態', '整合性', '診断'];
  const keys = ['family', 'variant', 'file', 'size', 'presence', 'readiness', 'integrity', 'error'];
  const element = document.createElement('table'), head = element.createTHead(), hrow = head.insertRow();
  for (const label of headers) { const th = document.createElement('th'); th.scope = 'col'; th.textContent = label; hrow.append(th); }
  const body = element.createTBody();
  for (const row of rows) { const tr = body.insertRow(); keys.forEach((key, i) => { const td = tr.insertCell(); td.dataset.label = headers[i]; td.textContent = row[key]; }); }
  host.append(element);
}
export function mountInventory(document, fetchImpl) {
  const button = document.getElementById('refresh'), notice = document.getElementById('notice'), observed = document.getElementById('observed'), cached = document.getElementById('cached'), configured = document.getElementById('configured');
  let pending = false;
  async function refresh() {
    if (pending) return;
    pending = true; button.disabled = true; notice.textContent = '観測中…';
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 5000);
    try {
      const response = await fetchImpl('/api/model-cache/status', { method: 'GET', cache: 'no-store', signal: controller.signal });
      if (!response.ok) throw new Error('Status HTTP failure');
      const snapshot = inventorySnapshot(await response.json());
      const ready = snapshot.rows.filter(row => row.cached);
      notice.textContent = snapshot.invalid ? `一部の状態が不正です（${snapshot.invalid}件）。そのファイルは保存・検証済みとして数えていません。` : `保存・検証済み ${ready.length}件 ／ 設定済み ${snapshot.rows.length}件`;
      observed.textContent = `サーバー観測時刻: ${new Date(snapshot.observedAt).toLocaleString('ja-JP')} ／ 受信時刻: ${new Date().toLocaleString('ja-JP')}`;
      table(document, cached, ready, snapshot.invalid ? '検証済みと確認できるファイルはありません。一部の状態は不明です。' : 'この観測では保存・検証済みファイルはありません。');
      table(document, configured, snapshot.rows.filter(row => !row.cached), '未準備の登録ファイルはありません。');
    } catch {
      notice.textContent = '取得不能：キャッシュAPIが利用できません。保存ファイル数・状態は不明です。';
      observed.textContent = `取得試行時刻: ${new Date().toLocaleString('ja-JP')}`;
      cached.replaceChildren(); configured.replaceChildren();
    } finally { clearTimeout(timer); pending = false; button.disabled = false; }
  }
  button.addEventListener('click', refresh);
  void refresh();
  return refresh;
}
