const API_BASE = "http://localhost:8000";   

// Check compression availability
let COMPRESSION_AVAILABLE = false;

// Check features on load
async function checkFeatures() {
  try {
    const resp = await fetch(`${API_BASE}/`);
    const data = await resp.json();
    COMPRESSION_AVAILABLE = data.compression_available || false;
    console.log('[DEBUG] Compression available:', COMPRESSION_AVAILABLE);
  } catch (error) {
    console.error('Failed to check features:', error);
  }
}

// Session management
let sessionId = localStorage.getItem("sessionId") || null;
let useAdvancedPrompting = false;
let selectedTechnique = 'auto';

function setSelectedTechnique(technique) {
  selectedTechnique = technique;
  console.log('Selected technique:', selectedTechnique);
}
let sessionHasFiles = false; // Track if current session has files

// DOM elements
const messagesDiv = document.getElementById('messages');
const input = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const clearBtn = document.getElementById('clear-chat');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
const suggestionChips = document.querySelectorAll('.suggestion-chip');
const settingsBtn = document.getElementById('settings-btn');
const advancedToggle = document.getElementById('advanced-toggle');
const newChatBtn = document.getElementById('new-chat-btn');
const saveSessionBtn = document.getElementById('save-session-btn');
const sessionsList = document.getElementById('sessions-list');
const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
const attachmentBtn = document.getElementById('attachment-btn');
const attachmentDropdown = document.getElementById('attachment-dropdown');
const attachmentOptions = document.querySelectorAll('.attachment-option');



// Load available prompting techniques as radio buttons
async function loadPromptingTechniques() {
  try {
    const resp = await fetch(`${API_BASE}/prompting/techniques`);
    const data = await resp.json();
    
    const radioGroupDiv = document.getElementById('technique-radio-group');
    if (radioGroupDiv) {
      radioGroupDiv.innerHTML = '';
      
      // Add an "Auto-Detect" option first
      const autoWrapper = document.createElement('div');
      autoWrapper.className = 'radio-option';
      
      const autoRadio = document.createElement('input');
      autoRadio.type = 'radio';
      autoRadio.id = 'technique-auto';
      autoRadio.name = 'technique';
      autoRadio.value = 'auto';
      autoRadio.checked = true; // Default selection
      autoRadio.addEventListener('change', () => setSelectedTechnique('auto'));
      
      const autoLabel = document.createElement('label');
      autoLabel.htmlFor = 'technique-auto';
      autoLabel.textContent = 'Auto-Detect';
      
      autoWrapper.appendChild(autoRadio);
      autoWrapper.appendChild(autoLabel);
      radioGroupDiv.appendChild(autoWrapper);
      
      // Add technique options with original names
      const techniqueNames = {
        'contrastive': 'contrastive',
        'few_shot': 'few_shot',
        'react': 'react',
        'auto_cot': 'auto_cot',
        'program_of_thought': 'program_of_thought'
      };
      
      data.techniques.forEach(technique => {
        const wrapper = document.createElement('div');
        wrapper.className = 'radio-option';
        
        const radio = document.createElement('input');
        radio.type = 'radio';
        radio.id = `technique-${technique.name}`;
        radio.name = 'technique';
        radio.value = technique.name;
        radio.addEventListener('change', () => setSelectedTechnique(technique.name));
        
        const label = document.createElement('label');
        label.htmlFor = `technique-${technique.name}`;
        label.textContent = techniqueNames[technique.name] || technique.name;
        
        wrapper.appendChild(radio);
        wrapper.appendChild(label);
        radioGroupDiv.appendChild(wrapper);
      });
    }
  } catch (error) {
    console.error('Failed to load prompting techniques:', error);
  }
}

// Helper: Render message
function addMessage(text, sender, techniqueUsed = null) {
  const emptyState = document.querySelector('.empty-state');
  if (emptyState) emptyState.remove();

  const messageRow = document.createElement('div');
  messageRow.className = `message-row ${sender}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${sender}`;
  avatar.textContent = sender === 'user' ? 'U' : 'G';
  messageRow.appendChild(avatar);

  const bubble = document.createElement('div');
  bubble.className = `bubble ${sender}`;
  bubble.innerHTML = marked.parse(text);

  bubble.querySelectorAll('pre code').forEach((block) => {
    hljs.highlightElement(block);
  });

  // Timestamp
  const timestamp = document.createElement('div');
  timestamp.className = 'timestamp';
  timestamp.textContent = (new Date()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  bubble.appendChild(timestamp);

  // Action buttons
  if (sender === 'assistant') {
    const actions = document.createElement('div');
    actions.className = 'ai-actions';
    
    const copyBtn = document.createElement('button');
    copyBtn.textContent = 'Copy';
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(text);
      copyBtn.textContent = 'Copied!';
      setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1200);
    };
    actions.appendChild(copyBtn);
    
    // Show technique used if available
    if (techniqueUsed && techniqueUsed !== 'standard') {
      const techniqueBtn = document.createElement('button');
      techniqueBtn.textContent = `✨ ${techniqueUsed}`;
      techniqueBtn.style.backgroundColor = 'rgba(0, 230, 208, 0.2)';
      techniqueBtn.style.color = 'var(--primary-color)';
      techniqueBtn.title = 'Advanced prompting technique used';
      actions.appendChild(techniqueBtn);
    }
    
    bubble.appendChild(actions);
  }
  if (sender === 'user') {
    const actions = document.createElement('div');
    actions.className = 'user-actions';
    const editBtn = document.createElement('button');
    editBtn.textContent = 'Edit';
    editBtn.onclick = () => {
      input.value = text;
      input.focus();
      autoResize();
    };
    actions.appendChild(editBtn);
    const repeatBtn = document.createElement('button');
    repeatBtn.textContent = 'Repeat';
    repeatBtn.onclick = () => {
      handleSend(text);
    };
    actions.appendChild(repeatBtn);
    bubble.appendChild(actions);
  }

  messageRow.appendChild(bubble);
  messagesDiv.appendChild(messageRow);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Typing indicator
function addTypingIndicator() {
  if (document.getElementById('typing-ind')) return;
  const typing = document.createElement('div');
  typing.id = 'typing-ind';
  typing.className = 'typing-indicator';
  typing.innerHTML = `
    <span class="typing-text">Generating response</span>
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
  `;
  messagesDiv.appendChild(typing);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
function removeTypingIndicator() {
  const typing = document.getElementById('typing-ind');
  if (typing) typing.remove();
}

// Auto-resize textarea
function autoResize() {
  input.style.height = 'auto';
  input.style.height = (input.scrollHeight) + 'px';
}
input.addEventListener('input', autoResize);

// Sidebar toggle & suggestion chips
sidebarToggle.addEventListener('click', () => {
  sidebar.classList.toggle('visible');
});
document.addEventListener('click', (e) => {
  if (sidebar.classList.contains('visible') && !sidebar.contains(e.target) && e.target !== sidebarToggle) {
    sidebar.classList.remove('visible');
  }
});
suggestionChips.forEach(chip => {
  chip.addEventListener('click', () => {
    input.value = chip.textContent;
    input.focus();
    autoResize();
  });
});

// Settings toggle
settingsBtn.addEventListener('click', () => {
  const advancedPanel = document.getElementById("advanced-prompting-group");
  const isHidden = advancedPanel.style.display === "none";
  
  // Toggle advanced panel
  advancedPanel.style.display = isHidden ? "block" : "none";
  
  if (isHidden) {
    loadPromptingTechniques(); // Load techniques when panel opens
  }
});


// Advanced prompting toggle
advancedToggle.addEventListener('change', () => {
  useAdvancedPrompting = advancedToggle.checked;
  const advancedSettings = document.getElementById('advanced-settings');
  advancedSettings.style.display = useAdvancedPrompting ? 'block' : 'none';
  
  // If disabled, reset technique to auto
    if (!useAdvancedPrompting) {
      selectedTechnique = 'auto';
      // Reset radio button selection to "Auto-Detect"
      const autoRadio = document.getElementById('technique-auto');
      if (autoRadio) {
        autoRadio.checked = true;
      }
    }
});


// Radio buttons for techniques are handled in loadPromptingTechniques function
// Each radio button gets its own event listener when created





// New chat: new session, clear chat
newChatBtn.addEventListener('click', async () => {
  const fd = new FormData();
  const resp = await fetch(`${API_BASE}/sessions`, { method: "POST", body: fd });
  const data = await resp.json();
  sessionId = data.id;
  localStorage.setItem("sessionId", sessionId);
  sessionHasFiles = false; // New session has no files
  clearChatUI();
});

// Clear chat UI only
function clearChatUI() {
  messagesDiv.innerHTML = '';
  const emptyState = document.createElement('div');
  emptyState.className = 'empty-state';
  emptyState.innerHTML = `
    <div class="empty-icon">💬</div>
    <div class="empty-title">Coding AI Assistant</div>
    <p class="empty-description">
      Ask questions about Python, get code explanations, debugging help, or discuss programming concepts.
    </p>
    <div class="suggestion-chips">
      <div class="suggestion-chip">How to use list comprehension?</div>
      <div class="suggestion-chip">Explain decorators in Python</div>
      <div class="suggestion-chip">Help with async/await syntax</div>
      <div class="suggestion-chip">Debug my error message</div>
    </div>
  `;
  messagesDiv.appendChild(emptyState);
  document.querySelectorAll('.suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      input.value = chip.textContent;
      input.focus();
      autoResize();
    });
  });
}

// Clear chat (also delete from backend)
clearBtn.onclick = async () => {
  if (sessionId) {
    await fetch(`${API_BASE}/history/${sessionId}`, { method: "DELETE" });
  }
  clearChatUI();
};

// Handle send (text message)
async function handleSend(rawInput) {
  const question = (rawInput !== undefined ? rawInput : input.value.trim());
  console.log('[DEBUG] handleSend called with question:', question);
  console.log('[DEBUG] useAdvancedPrompting:', useAdvancedPrompting);
  console.log('[DEBUG] selectedTechnique:', selectedTechnique);
  
  if (!question) {
    console.log('[DEBUG] No question provided, returning');
    return;
  }

  // Check if we have files attached (new files) or if session already has files
  const hasNewFiles = window.lastUploadedFile || document.querySelector('.file-preview');
  console.log('[DEBUG] Has new files attached:', hasNewFiles);
  console.log('[DEBUG] Session has files:', sessionHasFiles);
  console.log('[DEBUG] window.lastUploadedFile:', window.lastUploadedFile);
  console.log('[DEBUG] file preview element:', document.querySelector('.file-preview'));
  
  if (hasNewFiles) {
    console.log('[DEBUG] Sending via RAG endpoint with new files');
    await sendRAGMessage(question);
    return;
  }
  
  // If session has files but no new files, use regular chat which will access session files
  if (sessionHasFiles) {
    console.log('[DEBUG] Using regular chat endpoint - session has files');
    // Continue with regular chat endpoint - it will automatically use session files
  }
  
  // Determine which endpoint to use
  const endpoint = useAdvancedPrompting ? '/chat/advanced' : '/chat';
  console.log(`[DEBUG] Using endpoint: ${endpoint}`);

  addMessage(question, 'user');
  input.value = '';
  addTypingIndicator();

  // Use current or new session
  if (!sessionId) {
    const fd = new FormData();
    const resp = await fetch(`${API_BASE}/sessions`, { method: "POST", body: fd });
    const data = await resp.json();
    sessionId = data.id;
    localStorage.setItem("sessionId", sessionId);
  }

  const fd = new FormData();
  fd.append("session_id", sessionId);
  fd.append("question", question);
  if (useAdvancedPrompting && selectedTechnique) {
    fd.append("technique", selectedTechnique);
  }
  
  

  try {
    const resp = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      body: fd
    });
    const data = await resp.json();
    removeTypingIndicator();
    // Enhanced technique display with compression info
    let techniqueInfo = data.technique_used;
    if (data.compression_used) {
      techniqueInfo += ` (compressed)`;
    }
    
    if (data.prompt_length_chars && data.prompt_length_tokens) {
      showPromptInfo(data.prompt_length_chars, data.prompt_length_tokens);
    }
    
    addMessage(data.answer, "assistant", techniqueInfo);
    
    // If we got chunks back, it means RAG was used
    if (data.chunks && data.chunks.length > 0) {
      console.log("RAG context was used with", data.chunks.length, "chunks");
    }
  } catch (error) {
    removeTypingIndicator();
    addMessage("❌ Error: Could not get response. Please try again.", "assistant");
  }
}

// Handle send for RAG (with attachment)
async function sendRAGMessage(question) {
  console.log('[DEBUG] sendRAGMessage called with:', question);
  console.log('[DEBUG] sessionId:', sessionId);
  console.log('[DEBUG] lastUploadedFile:', window.lastUploadedFile);
  
  addMessage(question, 'user');
  input.value = '';
  addTypingIndicator();

  const fd = new FormData();
  fd.append("session_id", sessionId || "");
  fd.append("question", question);

  // Add advanced prompting parameters if enabled
  if (useAdvancedPrompting && selectedTechnique) {
    fd.append("technique", selectedTechnique);
  }

  // Attach files
  if (window.lastUploadedFile) {
    console.log('[DEBUG] Attaching file:', window.lastUploadedFile.name);
    fd.append("files", window.lastUploadedFile);
  } else {
    console.log('[DEBUG] No file to attach');
  }
  showUploadSpinner();
  try {
    const resp = await fetch(`${API_BASE}/rag`, {
      method: "POST",
      body: fd
    });
    const data = await resp.json();
    removeTypingIndicator();
    addMessage(data.answer, "assistant");
    
    // Update sessionId if we got a new one
    if (data.session_id) {
      sessionId = data.session_id;
      localStorage.setItem("sessionId", sessionId);
    }
    
    // Mark that this session now has files
    sessionHasFiles = true;
  } catch (error) {
    removeTypingIndicator();
    addMessage("❌ Error processing file. Please try again.", "assistant");
    console.error('RAG error:', error);
  }
  hideUploadSpinner();
  // Clean up file attachment
  window.lastUploadedFile = null;
  removeFilePreview();
  input.dataset.fileAttached = "";
}

// Send button
sendBtn.onclick = () => {
  handleSend();
};
input.addEventListener('keydown', (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  } else {
    autoResize();
  }
});

// Attachment dropdown functionality
attachmentBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  attachmentDropdown.classList.toggle('show');
});
document.addEventListener('click', () => {
  attachmentDropdown.classList.remove('show');
});
attachmentDropdown.addEventListener('click', (e) => {
  e.stopPropagation();
});
attachmentOptions.forEach(option => {
  option.addEventListener('click', () => {
    const type = option.getAttribute('data-type');
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.style.display = 'none';
    // Don't modify input value in onchange anymore
    if (type === 'photo-video') fileInput.accept = 'image/*,video/*';
    else if (type === 'document') fileInput.accept = '.pdf,.txt,.zip';
    else if (type === 'code') fileInput.accept = '.py,.js,.jsx,.ts,.tsx,.c,.h,.cpp,.hpp,.cc,.cxx,.java,.cs,.php,.rb,.go,.rs,.swift,.kt,.scala,.r,.R,.m,.pl,.pm,.lua,.sh,.bash,.zsh,.ps1,.bat,.cmd,.sql,.html,.htm,.xml,.css,.scss,.sass,.less,.json,.yaml,.yml,.toml,.md,.markdown,.vue,.dart,.elm,.ex,.exs,.erl,.hrl,.fs,.fsx,.hs,.jl,.nim,.pas,.pp,.vb,.vbs,.asm,.s,.clj,.cljs,.coffee,.groovy,.tcl,.dockerfile,.gitignore,.gitconfig,.makefile,.mk,.cmake';
    document.body.appendChild(fileInput);
    fileInput.click();

    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length > 0) {
        window.lastUploadedFile = fileInput.files[0];
        input.dataset.fileAttached = "true";
        
        // Create file preview
        showFilePreview(fileInput.files[0]);
        
        attachmentDropdown.classList.remove('show');
        document.body.removeChild(fileInput);
      }
    });
  });
});

function showPromptInfo(charCount, tokenCount) {
  const infoDiv = document.createElement('div');
  infoDiv.className = 'prompt-info-notification';
  infoDiv.innerHTML = `
    <strong>Prompt Sent</strong><br>
    Characters: ${charCount} | Tokens: ~${tokenCount} (estimated)
  `;
  
  document.body.appendChild(infoDiv);
  
  setTimeout(() => {
    infoDiv.classList.add('show');
  }, 100);
  
  setTimeout(() => {
    infoDiv.classList.remove('show');
    setTimeout(() => {
      infoDiv.remove();
    }, 500);
  }, 4000);
}

// Session management functions
async function loadSessions() {
  try {
    const resp = await fetch(`${API_BASE}/sessions`);
    const sessions = await resp.json();
    
    sessionsList.innerHTML = '';
    sessions.forEach(session => {
      const sessionDiv = document.createElement('div');
      sessionDiv.className = 'session-item';
      sessionDiv.dataset.sessionId = session.id;
      
      // Truncate title if too long
      const title = session.title.length > 30 ? session.title.substring(0, 30) + '...' : session.title;
      
      sessionDiv.innerHTML = `
        <div class="session-title">${title}</div>
        <div class="session-meta">
          <span>${new Date(session.created_at).toLocaleDateString()}</span>
          <span class="session-files-count">📁 ${session.file_count || 0}</span>
        </div>
        <button class="session-delete-btn" onclick="deleteSession('${session.id}', event)">×</button>
      `;
      
      // Mark current session as active
      if (session.id === sessionId) {
        sessionDiv.classList.add('active');
      }
      
      sessionDiv.addEventListener('click', () => loadSession(session.id));
      sessionsList.appendChild(sessionDiv);
    });
  } catch (error) {
    console.error('Failed to load sessions:', error);
  }
}

async function loadSession(newSessionId) {
  sessionId = newSessionId;
  localStorage.setItem('sessionId', sessionId);
  
  // Clear current chat
  messagesDiv.innerHTML = '';
  
  // Check if this session has files
  try {
    const sessionsResp = await fetch(`${API_BASE}/sessions`);
    if (sessionsResp.ok) {
      const sessions = await sessionsResp.json();
      const currentSession = sessions.find(s => s.id === sessionId);
      sessionHasFiles = currentSession && currentSession.file_count > 0;
      console.log('[DEBUG] Session', sessionId, 'has files:', sessionHasFiles);
    }
  } catch (error) {
    console.error('Failed to check session files:', error);
    sessionHasFiles = false;
  }
  
  // Load session history
  try {
    const resp = await fetch(`${API_BASE}/history/${sessionId}`);
    if (resp.ok) {
      const history = await resp.json();
      history.forEach(msg => {
        addMessage(msg.content, msg.role === 'user' ? 'user' : 'assistant');
      });
    }
  } catch (error) {
    console.error('Failed to load session history:', error);
  }
  
  // Update UI
  loadSessions(); // Refresh session list to update active state
}

async function saveCurrentSession() {
  if (!sessionId) {
    alert('No active session to save');
    return;
  }
  
  // Get first user message as default title
  const messages = messagesDiv.querySelectorAll('.message-row.user');
  if (messages.length === 0) {
    alert('No messages to save');
    return;
  }
  
  const firstMessage = messages[0].querySelector('.bubble').textContent;
  let defaultTitle = firstMessage.replace(/^Attached file:.*?\./g, '').trim();
  defaultTitle = defaultTitle.length > 50 ? defaultTitle.substring(0, 50) + '...' : defaultTitle;
  
  if (!defaultTitle) {
    defaultTitle = 'Untitled Session';
  }
  
  // Prompt user for custom name
  const customTitle = prompt('Enter a name for this session:', defaultTitle);
  if (customTitle === null) {
    return; // User cancelled
  }
  
  const title = customTitle.trim() || defaultTitle;
  
  try {
    const fd = new FormData();
    fd.append('title', title);
    
    const resp = await fetch(`${API_BASE}/sessions/${sessionId}/save`, {
      method: 'POST',
      body: fd
    });
    
    if (resp.ok) {
      alert('Session saved successfully!');
      loadSessions(); // Refresh session list
    } else {
      alert('Failed to save session');
    }
  } catch (error) {
    console.error('Failed to save session:', error);
    alert('Failed to save session');
  }
}

async function deleteSession(sessionIdToDelete, event) {
  event.stopPropagation(); // Prevent session from being loaded
  
  if (!confirm('Are you sure you want to delete this session? This will also delete all associated files.')) {
    return;
  }
  
  try {
    const resp = await fetch(`${API_BASE}/sessions/${sessionIdToDelete}`, {
      method: 'DELETE'
    });
    
    if (resp.ok) {
      // If we deleted the current session, create a new one
      if (sessionIdToDelete === sessionId) {
        const fd = new FormData();
        const newResp = await fetch(`${API_BASE}/sessions`, { method: 'POST', body: fd });
        const newData = await newResp.json();
        sessionId = newData.id;
        localStorage.setItem('sessionId', sessionId);
        clearChatUI();
      }
      
      loadSessions(); // Refresh session list
    } else {
      alert('Failed to delete session');
    }
  } catch (error) {
    console.error('Failed to delete session:', error);
    alert('Failed to delete session');
  }
}

// File preview functions
function showFilePreview(file) {
  // Remove any existing preview
  const existingPreview = document.querySelector('.file-preview');
  if (existingPreview) {
    existingPreview.remove();
  }
  
  // Create file preview element
  const preview = document.createElement('div');
  preview.className = 'file-preview';
  
  // Get file extension for icon
  const extension = file.name.split('.').pop().toUpperCase();
  const fileSize = formatFileSize(file.size);
  
  preview.innerHTML = `
    <div class="file-preview-icon">${extension.charAt(0)}</div>
    <div class="file-preview-info">
      <div class="file-preview-name">${file.name}</div>
      <div class="file-preview-size">${fileSize}</div>
    </div>
    <button class="file-remove-btn" onclick="removeFilePreview()" title="Remove file">×</button>
  `;
  
  // Insert before input wrapper
  const inputWrapper = document.querySelector('.input-wrapper');
  inputWrapper.parentNode.insertBefore(preview, inputWrapper);
}

function removeFilePreview() {
  const preview = document.querySelector('.file-preview');
  if (preview) {
    preview.remove();
  }
  window.lastUploadedFile = null;
  input.dataset.fileAttached = "";
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// Sidebar toggle functionality
sidebarToggleBtn.addEventListener('click', () => {
  sidebar.classList.toggle('collapsed');
});

// Save session button
saveSessionBtn.addEventListener('click', saveCurrentSession);
function showUploadSpinner() {
  let existing = document.getElementById('upload-spinner');
  if (existing) return;

  const spinner = document.createElement('div');
  spinner.id = 'upload-spinner';
  spinner.innerHTML = `
    <div class="spinner-overlay">
      <div class="spinner-box">
        <div class="loader"></div>
        <div class="spinner-text">Uploading and processing file...</div>
      </div>
    </div>
  `;
  document.body.appendChild(spinner);
}

function hideUploadSpinner() {
  const spinner = document.getElementById('upload-spinner');
  if (spinner) spinner.remove();
}


// On load: restore session and history
window.addEventListener('DOMContentLoaded', async () => {
  // Check available features first
  await checkFeatures();
  if (!sessionId) {
    // create new session
    const fd = new FormData();
    const resp = await fetch(`${API_BASE}/sessions`, { method: "POST", body: fd });
    const data = await resp.json();
    sessionId = data.id;
    localStorage.setItem("sessionId", sessionId);
    sessionHasFiles = false; // New session has no files
  }
  
  // Load sessions list
  await loadSessions();
  
  // Check if current session has files
  try {
    const sessionsResp = await fetch(`${API_BASE}/sessions`);
    if (sessionsResp.ok) {
      const sessions = await sessionsResp.json();
      const currentSession = sessions.find(s => s.id === sessionId);
      sessionHasFiles = currentSession && currentSession.file_count > 0;
      console.log('[DEBUG] Current session has files:', sessionHasFiles);
    }
  } catch (error) {
    console.error('Failed to check session files:', error);
    sessionHasFiles = false;
  }
  
  // load history
  const resp = await fetch(`${API_BASE}/history/${sessionId}`);
  if (resp.ok) {
    const history = await resp.json();
    messagesDiv.innerHTML = '';
    history.forEach(msg => {
      addMessage(msg.content, msg.role === "user" ? "user" : "assistant");
    });
  }
  setTimeout(() => {
    input.focus();
  }, 100);
});
