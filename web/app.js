const $ = id => document.getElementById(id);
let selectedFile, originalUrl, jsonUrl, result, busy = false;
const names = ['Red 5', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
const tileLabel = tile => {
  const code = tile.tile;
  if (/^[0-9][mps]$/.test(code)) {
    const suit = { m: code[0] === '1' ? 'character' : 'characters', p: code[0] === '1' ? 'dot' : 'dots', s: 'bamboo' }[code[1]];
    return `${names[Number(code[0])]} ${suit}`;
  }
  return tile.name || code;
};
const hasModelConfidence = tile => typeof tile.confidence === 'number' && Number.isFinite(tile.confidence)
  && tile.confidence >= 0 && tile.confidence <= 1;
const confidenceLabel = tile => hasModelConfidence(tile) ? `${(tile.confidence * 100).toFixed(1)}% model confidence`
  : 'Model confidence unavailable';
const setStatus = (text, error = false) => { $('status').textContent = text; $('status').classList.toggle('error', error); };
function showImage(annotated) {
  $('preview-image').src = annotated ? result.annotated_image : originalUrl;
  $('preview-image').alt = annotated ? 'Mahjong image with detected tiles marked' : 'Uploaded Mahjong image';
  $('original-tab').setAttribute('aria-pressed', String(!annotated));
  $('annotated-tab').setAttribute('aria-pressed', String(annotated));
}
async function selectFile(file) {
  if (busy || !file) return;
  const knownType = ['image/jpeg', 'image/png', 'image/webp', 'image/heif', 'image/heic', 'image/heif-sequence', 'image/heic-sequence'].includes(file.type);
  if ((!knownType && !/\.(jpe?g|png|webp|heif|heic)$/i.test(file.name)) || file.size > 50 * 1024 * 1024) {
    setStatus('Choose a JPG, PNG, WebP, HEIF, or HEIC image up to 50 MB.', true); return;
  }
  resetChat();
  busy = true; selectedFile = null; result = null; originalUrl = null;
  $('file').disabled = true; $('analyze').disabled = true;
  $('results').hidden = true; $('annotated-tab').disabled = true;
  $('preview-image').hidden = true; $('empty').hidden = false;
  $('empty').querySelector('h3').textContent = 'Preparing photo…';
  $('empty').querySelector('p').textContent = 'Converting and resizing on this computer.';
  $('filename').textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  if (jsonUrl) URL.revokeObjectURL(jsonUrl);
  setStatus('Preparing your photo…');
  try {
    const response = await fetch('/api/prepare', { method: 'POST', body: file });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not prepare this photo.');
    originalUrl = data.original_image; selectedFile = file;
    $('upload-controls').open = false;
    $('empty').hidden = true; $('preview-image').hidden = false;
    showImage(false); $('analyze').disabled = false;
    setStatus(`Ready to identify. Photo prepared at ${data.width} × ${data.height}.`);
  } catch (error) {
    $('empty').querySelector('h3').textContent = 'Could not open photo';
    $('empty').querySelector('p').textContent = 'Choose another file to try again.';
    setStatus(error.message || 'Could not connect to the app.', true);
  } finally { busy = false; $('file').disabled = false; }
}
$('file').addEventListener('change', event => selectFile(event.target.files[0]));
for (const type of ['dragenter', 'dragover']) $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.add('dragging'); });
for (const type of ['dragleave', 'drop']) $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.remove('dragging'); });
$('dropzone').addEventListener('drop', event => selectFile(event.dataTransfer.files[0]));
window.addEventListener('dragover', event => event.preventDefault());
window.addEventListener('drop', event => event.preventDefault());
$('model').addEventListener('change', () => {
  const clip = $('model').value === 'clip';
  $('detector-controls').hidden = clip;
  $('model-help').textContent = clip
    ? 'Ranks 42 tile labels for the whole photo. Use one tile per image. No detector or reference corrections.'
    : 'Finds tile locations and predicts each tile.';
  $('results').hidden = true; $('annotated-tab').disabled = true;
  result = null;
  if (originalUrl) showImage(false);
  setStatus(selectedFile ? 'Model changed. Identify again to compare.' : 'Choose an image to get started.');
});
$('confidence').addEventListener('input', () => {
  $('confidence-value').textContent = `${$('confidence').value}%`;
  if (result) setStatus('Threshold changed. Detect again to update the results.');
});
$('original-tab').onclick = () => { if (originalUrl) showImage(false); };
$('annotated-tab').onclick = () => { if (result) showImage(true); };
$('analyze').onclick = async () => {
  if (!selectedFile || busy) return;
  busy = true;
  for (const id of ['file', 'confidence', 'analyze', 'model']) $(id).disabled = true;
  $('analyze').textContent = 'Detecting…'; setStatus($('model').value === 'clip' ? 'Running CLIP… The first run also loads the model.' : 'Looking for tiles in your image…');
  $('results').hidden = true; $('annotated-tab').disabled = true; showImage(false);
  try {
    const response = await fetch(`/api/detect?model=${$('model').value}&confidence=${Number($('confidence').value) / 100}`, { method: 'POST', headers: { 'Content-Type': selectedFile.type }, body: selectedFile });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Detection failed. Please try again.');
    result = data;
    originalUrl = data.original_image;
    const isClip = data.analysis_type === 'classification';
    const predictions = isClip ? data.predictions.slice(0, 5) : data.detections;
    const total = predictions.length;
    $('result-model').textContent = data.model + (isClip ? ' · TOP 5 LABELS' : ' · DETECTIONS');
    $('score-note').textContent = isClip ? 'CLIP scores are relative softmax scores across 42 candidate labels, not calibrated accuracy or detector confidence. It always ranks labels, even if no tile is present.' : 'Scores indicate model confidence, not the probability that a prediction is correct.';
    const groups = new Map();
    for (const tile of predictions) groups.set(tileLabel(tile), (groups.get(tileLabel(tile)) || 0) + 1);
    $('result-title').textContent = isClip || total === 1 ? tileLabel(predictions[0]) : total ? `${total} possible tiles` : 'No tiles identified';
    $('result-description').textContent = isClip ? 'Top whole-image match. The entries below are alternative labels, not separate detected tiles.' : total === 1
      ? `1 tile detected · ${predictions[0].tile} · Check the crop below to confirm the match.`
      : total ? Array.from(groups, ([name, count]) => `${count} × ${name}`).join(' · ')
      : 'The model did not find a tile above the selected threshold. Try a clearer photo with the tile faces visible.';
    const sourceImage = new Image(); sourceImage.src = originalUrl;
    await sourceImage.decode();
    $('timing').textContent = `${data.original_width} × ${data.original_height} original → ${data.width} × ${data.height} processed · ${(data.elapsed_ms / 1000).toFixed(2)} s`;
    $('tiles').replaceChildren();
    for (const [index, tile] of predictions.entries()) {
      const card = document.createElement('div'); card.className = 'tile-card';
      if (!isClip) {
      const crop = document.createElement('canvas'); crop.className = 'tile-crop'; crop.width = 160; crop.height = 160;
      crop.setAttribute('role', 'img'); crop.setAttribute('aria-label', `Photo crop for prediction ${index + 1}`);
      const [x1, y1, x2, y2] = tile.xyxy;
      const scale = Math.min(144 / (x2 - x1), 144 / (y2 - y1));
      const width = (x2 - x1) * scale, height = (y2 - y1) * scale;
      crop.getContext('2d').drawImage(sourceImage, x1, y1, x2 - x1, y2 - y1, (160 - width) / 2, (160 - height) / 2, width, height);
      card.append(crop);
      } else {
        const rank = document.createElement('span'); rank.className = 'rank'; rank.textContent = index + 1; card.append(rank);
      }
      const details = document.createElement('div'); details.className = 'tile-detail';
      const title = document.createElement('strong'); title.textContent = tileLabel(tile);
      const code = document.createElement('p'); code.textContent = isClip ? tile.tile : `Tile ${index + 1} · ${tile.tile}`;
      const score = document.createElement('div'); score.className = 'score';
      const scoreLabel = document.createElement('span'); scoreLabel.textContent = isClip ? `${(tile.score * 100).toFixed(1)}% relative CLIP score` : confidenceLabel(tile);
      score.append(scoreLabel);
      if (hasModelConfidence(tile)) {
        const meter = document.createElement('meter'); meter.min = 0; meter.max = 1;
        meter.value = tile.confidence; meter.setAttribute('aria-label', 'Model confidence');
        score.append(meter);
      }
      details.append(code, title, score); card.append(details); $('tiles').append(card);
    }
    if (!predictions.length) {
      const message = document.createElement('p'); message.className = 'no-results';
      message.textContent = 'No tiles found at this threshold. Try a clearer, well-lit photo. An empty result does not identify the tile.';
      $('tiles').append(message);
    }
    $('download-image').hidden = isClip;
    if (!isClip) $('download-image').href = data.annotated_image;
    if (jsonUrl) URL.revokeObjectURL(jsonUrl);
    const { annotated_image, original_image, ...report } = data;
    jsonUrl = URL.createObjectURL(new Blob([JSON.stringify({ image: selectedFile.name, ...report }, null, 2)], { type: 'application/json' }));
    $('download-json').href = jsonUrl;
    $('results').hidden = false; $('annotated-tab').disabled = isClip;
    showImage(!isClip); setStatus('Done. Results are ready.');
  } catch (error) { result = null; setStatus(error.message || 'Could not connect. Check that the app is running.', true); }
  finally {
    busy = false;
    for (const id of ['file', 'confidence', 'analyze', 'model']) $(id).disabled = false;
    $('analyze').textContent = 'Identify tiles';
  }
};


let chatHistory = [], chatBusy = false;
function resetChat() {
  chatHistory = [];
  $('chat-messages').replaceChildren();
  $('chat-status').textContent = '';
  $('chat-photo').checked = false;
}
function chatBubble(role, text) {
  const bubble = document.createElement('div');
  bubble.className = `chat-message ${role}`;
  const label = document.createElement('strong'); label.textContent = role === 'user' ? 'You' : 'Gemini';
  const body = document.createElement('p'); body.textContent = text;
  bubble.append(label, body); $('chat-messages').append(bubble);
  bubble.scrollIntoView({ block: 'nearest' });
}
$('new-chat').onclick = () => { if (!chatBusy) resetChat(); };
async function sendChat(analyzePhoto = false) {
  if (chatBusy || busy) return;
  const question = analyzePhoto ? 'Identify the Mahjong tiles in this photo. List uncertain tiles and ask me to confirm or correct the hand before discussing scoring.' : $('chat-question').value.trim();
  if (!question) { $('chat-status').textContent = 'Enter a question first.'; return; }
  const includePhoto = analyzePhoto || $('chat-photo').checked;
  if (includePhoto && !originalUrl) { $('chat-status').textContent = 'Upload a photo first.'; return; }
  if (chatHistory.length >= 20) { $('chat-status').textContent = 'Start a new chat to continue after 10 exchanges.'; return; }
  chatBusy = true; busy = true;
  const controls = ['chat-send', 'ask-photo', 'new-chat', 'chat-question', 'chat-photo', 'file', 'model', 'analyze'];
  const disabled = controls.map(id => $(id).disabled);
  controls.forEach(id => $(id).disabled = true);
  $('chat-status').textContent = 'Gemini is thinking…';
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: question, history: chatHistory, ...(includePhoto ? { image: originalUrl } : {}) }),
      signal: AbortSignal.timeout(75000)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Gemini could not answer.');
    const displayedQuestion = question + (includePhoto ? '\n[Current photo included]' : '');
    chatBubble('user', displayedQuestion); chatBubble('assistant', data.reply);
    chatHistory.push({ role: 'user', text: displayedQuestion }, { role: 'assistant', text: data.reply });
    if (!analyzePhoto) $('chat-question').value = '';
    $('chat-status').textContent = `Answered by ${data.model}.`;
  } catch (error) {
    $('chat-status').textContent = error.name === 'TimeoutError' ? 'Gemini took too long. Please try again.' : error.message || 'Could not connect to the assistant.';
  } finally {
    chatBusy = false; busy = false;
    controls.forEach((id, index) => $(id).disabled = disabled[index]);
  }
}
$('chat-form').onsubmit = event => { event.preventDefault(); sendChat(); };
$('ask-photo').onclick = () => sendChat(true);
fetch('/api/assistant/status').then(response => response.json()).then(data => {
  $('assistant-config').textContent = data.configured
    ? `Ready · ${data.model}` : 'Setup: paste your Gemini API key into the project’s .env file. Then send a message; no restart needed.';
}).catch(() => { $('assistant-config').textContent = 'Start the Python server to use Gemini.'; });
