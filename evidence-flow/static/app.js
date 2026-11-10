const $ = (selector) => document.querySelector(selector);
const appState = {
  projects: [], projectId: localStorage.getItem('evidenceflow_project'),
  conversation: null, activeConclusion: 0, activeEvidence: null,
  source: null, screen: 'home', poller: null, visibleSuggestions: [],
};
const suggestions = [
  '头戴式智能设备如何综合头面部尺寸、接触压力、视野和视觉疲劳优化佩戴舒适度？',
  '面向老年人的智能穿戴产品，哪些人因和交互设计因素影响可用性、长期佩戴和接受度？',
  '可穿戴健康监测中，柔性传感器与多传感器融合分别能捕捉什么生理和运动信号，如何提高监测可靠性？',
  '不同传感方式（姿态、压力、毫米波）在人体行为识别中的优势与局限是什么？',
  '外骨骼康复设备在手部、腿部和腰部的助力机制、人体适配与舒适性评价有哪些异同？',
  '可穿戴个性化热管理如何结合生理信号监测、智能纤维和服装结构改善热舒适？',
  '智能眼镜如何结合眼动、面部动作、语音和触控实现免手操作，各方式适合什么场景？',
  '三维人体测量与参数化建模如何支持头戴、耳戴和腕戴产品的个性化适配？',
  '在视觉受限或高负荷场景中，多模态交互如何降低认知负荷并提高操作可靠性？',
  '柔性电子与智能织物如何兼顾可穿戴设备的传感性能、透气性和舒适性？',
  '眼镜镜腿夹持力与鼻托压力怎样共同影响佩戴舒适、滑移和长时间使用？',
  '头戴显示设备的重量、重心与转动惯量如何影响颈部负荷及舒适佩戴时长？',
  '不同地区的头面部测量差异，对护目镜、呼吸防护与头戴设备的尺码设计有何启示？',
  '三维扫描、参数化建模和 3D 打印如何改善个性化眼镜的贴合与舒适性？',
  'VR 头显面贴的几何适配、接触压力与热湿环境如何共同影响长期佩戴体验？',
  '运动或工业防护头盔怎样平衡冲击防护、视野、反应速度与接触压力？',
  '耳部三维形态、夹持力和局部压力如何影响耳机的适配与热舒适？',
  '下肢康复外骨骼的人机对位、腿部绑缚与助力控制如何影响安全性、舒适性和康复效果？',
  '表面肌电、接触压力与主观量表如何组合评估外骨骼的助力效果和穿戴舒适度？',
  'How do weight distribution, fit, and clamping force affect AR glasses comfort during prolonged wear?',
  'How can population differences in craniofacial measurements inform the fit of eyewear and protective headgear?',
  'Which 3D scanning, statistical modeling, and additive manufacturing methods improve personalized eyeglass fit?',
  'How do nose pad design, pressure thresholds, and temple forces contribute to eyeglass slippage and discomfort?',
  'How do HMD weight distribution, thermal load, and facial interface fit jointly affect physical workload and VR comfort?',
  'How do helmet fit and combined eye protection affect visual field, reaction time, and impact safety?',
  'Which hands-free input methods for smart glasses work best during walking or manual tasks, and why?',
  'How can ear shape measurements and local pressure thresholds guide comfortable earphone sizing and long-term wear?',
  'How do flexible textile and pressure sensors balance sensing accuracy, breathability, and comfort in wearable products?',
];
const suggestionIcons = ['10021-imgGroup26.svg', '10021-imgIcon.svg', '10021-imgIcon1.svg', '10021-imgGroup27.svg'];

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function short(value, length = 75) { const text = String(value ?? ''); return text.length > length ? text.slice(0, length) + '…' : text; }
function asset(name) { return `/static/assets/${name}?v=20261028`; }
function toast(message) { const node = $('#toast'); node.textContent = message; node.classList.add('show'); clearTimeout(toast.timer); toast.timer = setTimeout(() => node.classList.remove('show'), 4500); }
async function request(path, options = {}) {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json', ...(options.headers || {}) }, ...options });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.message || `请求失败 ${res.status}`);
  return data;
}
function currentProject() { return appState.projects.find(p => p.id === appState.projectId) || appState.projects[0]; }
function hideConversationMenu() { $('#conversation-context-menu')?.remove(); }
function showConversationMenu(event, conversationId) {
  event.preventDefault();
  hideConversationMenu();
  const menu = document.createElement('div');
  menu.id = 'conversation-context-menu';
  menu.className = 'conversation-context-menu';
  menu.setAttribute('role', 'menu');
  menu.innerHTML = '<button type="button" role="menuitem">删除对话</button>';
  document.body.appendChild(menu);
  const rect = menu.getBoundingClientRect();
  menu.style.left = `${Math.max(8, Math.min(event.clientX, window.innerWidth - rect.width - 8))}px`;
  menu.style.top = `${Math.max(8, Math.min(event.clientY, window.innerHeight - rect.height - 8))}px`;
  menu.querySelector('button').onclick = event => { event.stopPropagation(); hideConversationMenu(); deleteConversation(conversationId); };
  menu.querySelector('button').focus();
}
async function deleteConversation(conversationId) {
  const item = appState.projects.flatMap(project => project.conversations || []).find(c => c.id === conversationId);
  if (!item || !window.confirm(`确定删除这条研究对话吗？\n\n${short(item.question, 80)}\n\n此操作无法撤销。`)) return;
  try {
    await request(`/api/conversations/${encodeURIComponent(conversationId)}`, { method: 'DELETE' });
    if (appState.conversation?.id === conversationId) {
      clearInterval(appState.poller);
      appState.poller = null;
      appState.conversation = null;
      appState.activeConclusion = 0;
      appState.activeEvidence = null;
      appState.source = null;
      navigate('home');
      renderHome();
    }
    await loadProjects();
    toast('对话已删除');
  } catch (error) { toast(error.message); }
}
function navigate(screen, conversationId) {
  hideConversationMenu();
  appState.screen = screen;
  $('#home-screen').hidden = screen !== 'home';
  $('#analysis-screen').hidden = screen !== 'analysis';
  $('#final-screen').hidden = screen !== 'final';
  $('#global-header').classList.toggle('home-hidden', screen === 'home');
  $('#global-header').classList.toggle('research-header', screen !== 'home');
  $('#analysis-header-actions').hidden = screen === 'home';
  $('#header-verify').hidden = screen !== 'final';
  $('#header-final').hidden = screen !== 'analysis';
  $('#header-verify').classList.toggle('active', screen === 'analysis');
  $('#header-final').classList.toggle('active', screen === 'final');
  $('#home-nav').classList.toggle('active', screen === 'home');
  $('#projects-nav').classList.toggle('active', screen !== 'home');
  $('#sidebar').classList.remove('open');
  const hash = screen === 'home' ? '#/' : `#/${screen}/${conversationId}`;
  if (location.hash !== hash) history.replaceState(null, '', hash);
}
async function loadProjects() {
  appState.projects = await request('/api/projects');
  if (!appState.projects.some(p => p.id === appState.projectId)) appState.projectId = appState.projects[0]?.id || null;
  if (appState.projectId) localStorage.setItem('evidenceflow_project', appState.projectId);
  else localStorage.removeItem('evidenceflow_project');
  renderSidebar();
}
function renderSidebar() {
  hideConversationMenu();
  const search = $('#global-search-input').value.trim().toLowerCase();
  $('#projects-list').innerHTML = (!appState.projects.length ? '<div class="empty-project">首次提问后自动创建研究项目</div>' : '') + appState.projects.map(project => {
    const matches = (project.conversations || []).filter(c => !search || c.question.toLowerCase().includes(search));
    const open = project.id === appState.projectId;
    return `<div class="project-group"><button class="project-head ${open ? 'selected' : ''}" data-project="${project.id}"><span>${open ? '⌄' : '›'}</span><strong>${escapeHtml(project.name)}</strong><span>${project.conversations?.length || 0}</span></button>${open ? `<div>${matches.length ? matches.map(c => `<button class="project-chat ${appState.conversation?.id === c.id ? 'active' : ''}" data-conversation="${c.id}" title="${escapeHtml(c.question)}">${escapeHtml(short(c.question, 22))}</button>`).join('') : `<div class="empty-project">${search ? '没有匹配的对话' : '还没有研究对话'}</div>`}</div>` : ''}</div>`;
  }).join('') + '<button class="project-head new-project" id="new-project"><span>＋</span><strong>新建项目</strong></button>';
  document.querySelectorAll('[data-project]').forEach(node => node.onclick = () => {
    const id = node.dataset.project;
    appState.projectId = appState.projectId === id ? null : id;
    if (appState.projectId) localStorage.setItem('evidenceflow_project', appState.projectId);
    renderSidebar();
  });
  document.querySelectorAll('[data-conversation]').forEach(node => {
    node.onclick = () => openConversation(node.dataset.conversation);
    node.oncontextmenu = event => showConversationMenu(event, node.dataset.conversation);
    node.onkeydown = event => { if (event.key === 'ContextMenu' || (event.shiftKey && event.key === 'F10')) showConversationMenu(event, node.dataset.conversation); };
  });
  $('#new-project').onclick = showProjectModal;
}
function showProjectModal() { $('#project-modal').hidden = false; $('#project-name').value = ''; $('#project-name').focus(); }
function hideProjectModal() { $('#project-modal').hidden = true; }
function renderHome() {
  $('#home-screen').innerHTML = `<div class="home-hero"><div class="hero-text"><div class="eyebrow">从文献中发现答案，支持更可靠的设计研究</div><h1>在智能穿戴与人体工学文献库<br>中<em>提出研究问题</em></h1><p>基于「智能穿戴与人体工学」领域的中英文文献<br>通过 RAG 检索与多文献分析，获得有据可依的研究结论</p></div><img class="hero-art" src="${asset('10021-img1.png')}" alt="AR 眼镜与研究文献"></div><form id="question-form" class="question-composer"><textarea id="question-input" maxlength="500" placeholder="请输入你想探索的研究问题…" aria-label="研究问题"></textarea><div class="composer-bottom"><button type="button" class="composer-icon" id="attachment-btn" title="文献来自当前资料库"><img src="${asset('10021-imgAttachment.svg')}" alt=""></button><button type="button" class="source-chip selected" id="library-source"><img src="${asset('10021-imgLibraryIcon.svg')}" alt="">文献库 <img src="${asset('10021-imgLibraryArrow.svg')}" alt=""></button><button type="button" class="source-chip" id="web-source"><img src="${asset('10021-imgWebIcon.svg')}" alt="">联网搜索 <img src="${asset('10021-imgLibraryArrow.svg')}" alt=""></button><span class="counter" id="question-count">0/500</span><button type="submit" class="primary-button search-button"><img src="${asset('10021-imgSearchIcon.svg')}" alt="">开始检索与分析<img src="${asset('10021-imgGoArrow.svg')}" alt=""></button></div></form><div class="suggestions"><div class="section-heading">试试这些问题<button class="refresh" id="refresh-suggestions"><img src="${asset('10021-imgVector1.svg')}" alt="">换一批</button></div><div class="suggestion-grid" id="suggestion-grid"></div></div>`;
  $('#question-form').onsubmit = async event => { event.preventDefault(); await startQuestion($('#question-input').value); };
  $('#question-input').oninput = () => $('#question-count').textContent = `${$('#question-input').value.length}/500`;
  $('#question-input').onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('#question-form').requestSubmit(); } };
  $('#library-source').onclick = () => toast('已使用当前文献库进行检索');
  $('#web-source').onclick = () => toast('本轮研究使用本地文献库；联网搜索尚未接入此流程');
  $('#attachment-btn').onclick = () => toast('当前文献库已载入；可直接开始检索');
  $('#refresh-suggestions').onclick = renderSuggestions;
  renderSuggestions();
}
function renderSuggestions() {
  const previous = new Set(appState.visibleSuggestions);
  const candidates = suggestions.map((_, i) => i).filter(i => !previous.has(i));
  for (let i = candidates.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
  }
  appState.visibleSuggestions = candidates.slice(0, 4);
  $('#suggestion-grid').innerHTML = appState.visibleSuggestions.map((index, i) => `<button class="suggestion-card" data-suggestion="${index}" title="${escapeHtml(suggestions[index])}"><img class="icon" src="${asset(suggestionIcons[i])}" alt=""><span><strong>${escapeHtml(suggestions[index])}</strong></span><img class="arrow" src="${asset('10021-imgCardArrow.svg')}" alt=""></button>`).join('');
  document.querySelectorAll('[data-suggestion]').forEach(node => node.onclick = () => { $('#question-input').value = suggestions[Number(node.dataset.suggestion)]; $('#question-input').dispatchEvent(new Event('input')); $('#question-input').focus(); });
}
async function startQuestion(question) {
  const text = String(question || '').trim();
  if (text.length < 2) { toast('请输入研究问题'); return; }
  const project = currentProject();
  try {
    const conversation = await request('/api/conversations', { method: 'POST', body: JSON.stringify({ project_id: project?.id || null, question: text }) });
    appState.conversation = conversation;
    appState.projectId = conversation.project_id;
    appState.activeConclusion = 0;
    appState.activeEvidence = null;
    appState.source = null;
    await loadProjects();
    navigate('analysis', conversation.id);
    renderAnalysis();
    pollConversation(conversation.id);
  } catch (error) { toast(error.message); }
}
async function openConversation(id) {
  try {
    clearInterval(appState.poller);
    appState.conversation = await request(`/api/conversations/${id}`);
    appState.projectId = appState.conversation.project_id;
    localStorage.setItem('evidenceflow_project', appState.projectId);
    appState.activeConclusion = 0;
    appState.activeEvidence = null;
    appState.source = null;
    renderSidebar();
    navigate(appState.conversation.status === 'final' ? 'final' : 'analysis', id);
    if (appState.screen === 'final') renderFinal(); else renderAnalysis();
    if (['retrieving', 'analyzing'].includes(appState.conversation.status)) pollConversation(id);
  } catch (error) { toast(error.message); }
}
function pollConversation(id) {
  clearInterval(appState.poller);
  appState.poller = setInterval(async () => {
    if (appState.conversation?.id !== id) { clearInterval(appState.poller); return; }
    try {
      const next = await request(`/api/conversations/${id}`);
      if (appState.conversation?.id !== id) return;
      if (next.status !== appState.conversation.status) { appState.conversation = next; renderAnalysis(); renderSidebar(); }
      if (!['retrieving', 'analyzing'].includes(next.status)) clearInterval(appState.poller);
    } catch (error) { clearInterval(appState.poller); toast(error.message); }
  }, 2500);
}
function currentConclusion() { return appState.conversation?.state?.conclusions?.[appState.activeConclusion]; }
function candidateById(id) { return appState.conversation?.state?.candidates?.find(c => c.id === id); }
function evidenceCard(candidate, conclusion, index) {
  const selected = conclusion.selected_evidence_ids.includes(candidate.id);
  const thumbnail = candidate.images?.[0] ? `<img class="evidence-thumb" src="/api/images/${encodeURIComponent(candidate.images[0])}" alt="文献图片缩略图" loading="lazy">` : '';
  return `<div class="evidence-card ${selected ? 'selected' : ''}" data-preview="${candidate.id}" role="button" tabindex="0"><button class="check-box" data-select="${candidate.id}" title="${selected ? '取消选择' : '选择证据'}" aria-label="${selected ? '取消选择' : '选择证据'}">${selected ? `<img src="${asset('10022-imgCheck.svg')}" alt="">` : ''}</button><span class="evidence-number">${String(index+1).padStart(2,'0')}</span><span class="file-icon"><img src="${asset('10022-imgFile.svg')}" alt=""></span><span class="evidence-copy"><strong>${escapeHtml(candidate.source_file || candidate.title.split(' / ')[0] || '文献')}${candidate.page ? ` · 第 ${candidate.page} 页` : ''}</strong><span class="source-name">${escapeHtml(short(candidate.title, 85))}</span><p>${escapeHtml(short(candidate.excerpt, 150))}</p></span>${thumbnail}<span class="relevance">${selected ? '高相关' : '待核验'}</span><span class="view-chevron">›</span></div>`;
}
function renderAnalysis() {
  const c = appState.conversation;
  if (!c) return;
  const conclusions = c.state?.conclusions || [];
  const conclusion = currentConclusion();
  const busy = ['retrieving', 'analyzing'].includes(c.status);
  const status = busy ? `<div class="status-panel"><h2><span class="spinner"></span>${c.status === 'retrieving' ? '正在检索文献候选' : 'DeepSeek 正在筛选并归纳结论'}</h2><p>${c.status === 'retrieving' ? '正在向本地 RAG 服务请求足够的候选片段。' : `已取得 ${c.state?.candidates?.length || 0} 个候选片段，正在整理结论与证据关系。`}</p></div>` : c.status === 'error' ? `<div class="status-panel"><h2>分析未完成</h2><p>${escapeHtml(c.error || '请稍后重试')}</p><button class="ghost-button" id="retry-question">重新提问</button></div>` : !conclusions.length ? `<div class="status-panel"><h2>目前没有足够证据形成结论</h2><p>模型未从候选文献中找到可支持的结论。可以调整问题后重新提问。</p><button class="ghost-button" id="retry-question">重新提问</button></div>` : '';
  const evidenceIds = conclusion?.evidence_ids || [];
  $('#analysis-screen').innerHTML = `<div class="analysis-layout"><div class="analysis-left"><button class="back-button" id="back-home">‹ 返回研究问题</button><h1 class="analysis-title">${escapeHtml(c.question)}</h1><p class="analysis-subtitle">${busy ? '正在查找文献并生成结论…' : `${conclusions.length} 条研究结论 · ${c.state?.candidates?.length || 0} 个 RAG 候选片段`}</p>${conclusions.length ? `<div class="conclusion-tabs">${conclusions.map((x,i) => `<button class="conclusion-tab ${i === appState.activeConclusion ? 'active' : ''}" data-conclusion="${i}">${i+1}. ${escapeHtml(short(x.title, 23))}</button>`).join('')}</div><div class="conclusion-statement"><strong>${escapeHtml(conclusion.title)}</strong><p>${escapeHtml(conclusion.summary)}</p></div><div class="evidence-heading">相关证据 <small>请选择需要保留的原文证据</small></div><div class="evidence-list">${evidenceIds.map((id,i) => { const candidate = candidateById(id); return candidate ? evidenceCard(candidate, conclusion, i) : ''; }).join('')}</div><form class="followup-box" id="followup-form"><img src="${asset('10022-imgPaperclip.svg')}" alt=""><input id="followup-input" maxlength="500" placeholder="继续提问，或让 AI 进一步分析…"><button title="新建对话" aria-label="发送新问题"><img src="${asset('10022-imgSend.svg')}" alt=""></button></form>` : `<div class="evidence-list">${status}</div>`}</div><div class="analysis-right"><div class="source-preview"><div class="preview-toolbar"><img src="${asset('10022-imgFile.svg')}" alt=""><strong id="preview-name">${escapeHtml(appState.source?.name || 'Markdown 原文预览')}</strong><span id="preview-location"></span></div><div class="preview-document" id="preview-document"></div></div><button class="primary-button finalize-button" id="finalize-btn" ${!conclusions.length ? 'disabled' : ''}><img src="${asset('10022-imgSparkles.svg')}" alt="">生成最终结论<img src="${asset('10022-imgArrow.svg')}" alt=""></button></div></div>`;
  $('#back-home').onclick = () => { navigate('home'); renderHome(); };
  if ($('#retry-question')) $('#retry-question').onclick = () => { navigate('home'); renderHome(); $('#question-input').value = c.question; $('#question-input').dispatchEvent(new Event('input')); };
  document.querySelectorAll('[data-conclusion]').forEach(node => node.onclick = () => { appState.activeConclusion = Number(node.dataset.conclusion); appState.activeEvidence = null; appState.source = null; renderAnalysis(); });
  document.querySelectorAll('[data-preview]').forEach(node => { node.onclick = () => openSource(node.dataset.preview); node.onkeydown = event => { if (event.key === 'Enter') openSource(node.dataset.preview); }; });
  document.querySelectorAll('[data-select]').forEach(node => node.onclick = event => { event.stopPropagation(); toggleEvidence(node.dataset.select); });
  if ($('#followup-form')) $('#followup-form').onsubmit = event => { event.preventDefault(); startQuestion($('#followup-input').value); };
  $('#finalize-btn').onclick = finalize;
  renderSource();
  if (!appState.activeEvidence && evidenceIds.length) openSource(evidenceIds[0]);
}
async function toggleEvidence(id) {
  const conclusion = currentConclusion();
  if (!conclusion) return;
  const ids = new Set(conclusion.selected_evidence_ids);
  if (ids.has(id)) ids.delete(id); else ids.add(id);
  try {
    appState.conversation = await request(`/api/conversations/${appState.conversation.id}/selection`, { method: 'POST', body: JSON.stringify({ conclusion_id: conclusion.id, evidence_ids: [...ids] }) });
    renderAnalysis();
  } catch (error) { toast(error.message); }
}
async function openSource(id) {
  if (!appState.conversation) return;
  appState.activeEvidence = id;
  try {
    appState.source = await request(`/api/conversations/${appState.conversation.id}/source/${id}`);
    renderSource();
  } catch (error) { toast(error.message); }
}
function renderSource() {
  const container = $('#preview-document');
  if (!container) return;
  $('.preview-toolbar .pdf-controls')?.remove();
  const source = appState.source;
  if (!source) { container.innerHTML = '<div class="preview-empty">选择一条证据，在这里打开论文原文并定位片段</div>'; return; }
  $('#preview-name').textContent = source.name;
  $('#preview-location').textContent = source.pdf ? '' : source.located ? '已定位' : '仅有检索片段';
  container.replaceChildren();
  if (source.pdf) {
    const page = Math.min(Math.max(source.viewPage || source.page, 1), source.page_count);
    const zoom = source.zoom || 100;
    const controls = document.createElement('div');
    controls.className = 'pdf-controls';
    controls.innerHTML = `<div class="pdf-control-group"><button class="pdf-icon-button" data-pdf-step="-1" aria-label="上一页" title="上一页" ${page <= 1 ? 'disabled' : ''}>‹</button><label class="pdf-page-control"><input id="pdf-page-number" type="number" min="1" max="${source.page_count}" value="${page}" aria-label="PDF 页码"><span>/ ${source.page_count}</span></label><button class="pdf-icon-button" data-pdf-step="1" aria-label="下一页" title="下一页" ${page >= source.page_count ? 'disabled' : ''}>›</button></div><span class="pdf-toolbar-divider"></span><div class="pdf-control-group"><button class="pdf-icon-button" data-pdf-zoom="-20" aria-label="缩小" title="缩小">−</button><span class="pdf-zoom-label">${zoom}%</span><button class="pdf-icon-button" data-pdf-zoom="20" aria-label="放大" title="放大">＋</button></div><span class="pdf-toolbar-spacer"></span><button class="pdf-locate-button" data-pdf-locate="1">定位证据</button><a class="pdf-open-link" href="${source.pdf_url}#page=${page}" target="_blank" rel="noopener" aria-label="在新窗口打开 PDF 原件" title="在新窗口打开 PDF 原件">↗</a>`;
    controls.querySelectorAll('[data-pdf-step]').forEach(button => button.onclick = () => { source.viewPage = page + Number(button.dataset.pdfStep); renderSource(); });
    controls.querySelector('[data-pdf-locate]').onclick = () => { source.viewPage = source.page; renderSource(); };
    controls.querySelectorAll('[data-pdf-zoom]').forEach(button => button.onclick = () => { source.zoom = Math.min(200, Math.max(60, zoom + Number(button.dataset.pdfZoom))); renderSource(); });
    controls.querySelector('#pdf-page-number').onchange = event => { source.viewPage = Math.min(source.page_count, Math.max(1, Number(event.target.value) || page)); renderSource(); };
    const image = document.createElement('img');
    image.className = 'pdf-page-image';
    image.alt = `${source.name} 第 ${page} 页`;
    image.src = `/api/conversations/${encodeURIComponent(appState.conversation.id)}/pdf-page/${encodeURIComponent(source.evidence_id)}/${page}?render=3`;
    image.style.width = `${zoom}%`;
    const stage = document.createElement('div');
    stage.className = 'pdf-page-stage';
    stage.append(image);
    $('.preview-toolbar').append(controls);
    container.append(stage);
    container.scrollTop = 0;
    return;
  }
  const lines = source.markdown.split('\n');
  let offset = 0;
  let firstMark = null;
  for (const line of lines) {
    const node = document.createElement('div');
    const heading = line.match(/^(#{1,3})\s/);
    node.className = `md-line${heading ? ` heading h${heading[1].length}` : ''}`;
    const localStart = Math.max(0, (source.start ?? -1) - offset);
    const localEnd = Math.min(line.length, (source.end ?? -1) - offset);
    if (localEnd > localStart) {
      node.append(document.createTextNode(line.slice(0, localStart)));
      const mark = document.createElement('mark');
      mark.textContent = line.slice(localStart, localEnd);
      node.append(mark, document.createTextNode(line.slice(localEnd)));
      firstMark ||= mark;
    } else node.textContent = line || '\u00a0';
    container.append(node);
    offset += line.length + 1;
  }
  if (firstMark) requestAnimationFrame(() => firstMark.scrollIntoView({ block: 'center', behavior: 'smooth' }));
}
async function finalize() {
  const c = appState.conversation;
  if (!c) return;
  try { appState.conversation = await request(`/api/conversations/${c.id}/finalize`, { method: 'POST' }); navigate('final', c.id); renderFinal(); await loadProjects(); }
  catch (error) { toast(error.message); }
}
function renderFinal() {
  const c = appState.conversation;
  if (!c) return;
  const conclusions = c.state?.conclusions || [];
  const evidenceCount = conclusions.reduce((sum, x) => sum + x.selected_evidence_ids.length, 0);
  $('#final-screen').innerHTML = `<div class="final-breadcrumb">研究结论 / ${escapeHtml(currentProject()?.name || '研究项目')}</div><h1>研究结论</h1><h2>${escapeHtml(c.question)}</h2><p class="final-description">以下呈现本次研究的结论与用户在核验阶段保留的原文证据。</p><div class="final-summary"><span class="summary-icon">✓</span><span><strong>已生成最终研究结论</strong><p>包含 ${conclusions.length} 条研究发现 · ${evidenceCount} 条已确认原文证据可回查</p></span></div>${conclusions.map((conclusion,index) => `<article class="final-card"><span class="final-number">${index+1}</span><div><h3>${escapeHtml(conclusion.title)}</h3><p>${escapeHtml(conclusion.summary)}</p>${conclusion.selected_evidence_ids.length ? conclusion.selected_evidence_ids.map(id => { const candidate = candidateById(id); return candidate ? `<div class="final-evidence" data-final-evidence="${id}"><strong>${escapeHtml(candidate.source_file || candidate.title)} · ${id}</strong><span>“${escapeHtml(candidate.excerpt)}”</span></div>` : ''; }).join('') : '<p>此结论未保留证据；建议返回核验页选择原文片段。</p>'}</div><span class="confirmed">✓ 已确认</span></article>`).join('')}<div class="final-actions"><button class="ghost-button" id="back-analysis">‹ 返回证据核验</button><button class="primary-button" id="new-question">＋ 提出新问题</button></div>`;
  $('#back-analysis').onclick = () => { navigate('analysis', c.id); renderAnalysis(); };
  $('#new-question').onclick = () => { navigate('home'); renderHome(); };
  document.querySelectorAll('[data-final-evidence]').forEach(node => node.onclick = () => { const id = node.dataset.finalEvidence; const index = conclusions.findIndex(x => x.evidence_ids.includes(id)); appState.activeConclusion = Math.max(0, index); appState.activeEvidence = null; appState.source = null; navigate('analysis', c.id); renderAnalysis(); openSource(id); });
}
async function boot() {
  try { const match = location.hash.match(/^#\/(analysis|final)\/([a-f0-9]+)$/); await loadProjects(); renderHome(); navigate('home'); if (match) await openConversation(match[2]); }
  catch (error) { toast(error.message); }
}
$('#home-nav').onclick = () => { navigate('home'); renderHome(); };
$('#brand-home').onclick = () => { navigate('home'); renderHome(); };
$('#projects-nav').onclick = () => { const project = currentProject(); if (project) { appState.projectId = project.id; renderSidebar(); } else showProjectModal(); };
$('#library-nav').onclick = () => toast('文献库已由本地 RAG 服务索引，可在首页直接检索');
$('#settings-btn').onclick = async () => { const info = await request('/api/config'); toast(`RAG: ${info.rag_url} · DeepSeek: ${info.model} · ${info.key_configured ? '密钥已配置' : '密钥未配置'}`); };
$('#help-btn').onclick = () => toast('在首页输入研究问题，在核验页选择证据，最后查看完整结论。');
$('#notice-btn').onclick = () => toast('暂无新通知');
$('#mobile-menu').onclick = () => $('#sidebar').classList.toggle('open');
function setSidebarCollapsed(collapsed) {
  document.body.classList.toggle('sidebar-collapsed', collapsed);
  $('#sidebar-toggle').setAttribute('aria-label', collapsed ? '展开侧边栏' : '收起侧边栏');
  localStorage.setItem('evidenceflow_sidebar_collapsed', collapsed ? '1' : '0');
}
setSidebarCollapsed(localStorage.getItem('evidenceflow_sidebar_collapsed') === '1');
$('#sidebar-toggle').onclick = () => setSidebarCollapsed(true);
$('#sidebar-restore').onclick = () => setSidebarCollapsed(false);
$('#header-home').onclick = () => { navigate('home'); renderHome(); };
$('#header-verify').onclick = () => { if (!appState.conversation) return; navigate('analysis', appState.conversation.id); renderAnalysis(); };
$('#header-final').onclick = () => { if (appState.screen === 'final') return; finalize(); };
$('#global-search-input').oninput = renderSidebar;
document.addEventListener('pointerdown', event => { if (!event.target.closest('#conversation-context-menu')) hideConversationMenu(); });
document.addEventListener('keydown', event => { if (event.key === 'Escape') hideConversationMenu(); });
document.addEventListener('scroll', hideConversationMenu, true);
$('#cancel-project').onclick = hideProjectModal;
$('#project-modal').onclick = event => { if (event.target.id === 'project-modal') hideProjectModal(); };
$('#project-form').onsubmit = async event => { event.preventDefault(); try { const project = await request('/api/projects', { method: 'POST', body: JSON.stringify({ name: $('#project-name').value }) }); appState.projectId = project.id; await loadProjects(); hideProjectModal(); toast(`已创建项目「${project.name}」`); } catch (error) { toast(error.message); } };
boot();
