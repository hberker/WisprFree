// WisperFree tray app. Spawns the Python daemon (unless one is already
// running), shows a menu-bar/tray icon to toggle dictation, and opens
// the settings window. All traffic stays on 127.0.0.1.
const { app, Tray, Menu, BrowserWindow, nativeImage, shell } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

const API = process.env.WISPERFREE_API || 'http://127.0.0.1:8765';

let tray = null;
let settingsWindow = null;
let daemonProcess = null;
let dictating = false;

async function api(route, options = {}) {
  const response = await fetch(`${API}${route}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) throw new Error(`${route}: ${response.status}`);
  return response.json();
}

async function ensureDaemon() {
  try {
    await api('/api/status');
    return; // already running
  } catch {
    daemonProcess = spawn('wisperfree', [], {
      stdio: 'ignore',
      detached: false,
    });
    // Give it a moment to bind
    for (let i = 0; i < 20; i++) {
      await new Promise((r) => setTimeout(r, 500));
      try { await api('/api/status'); return; } catch { /* retry */ }
    }
  }
}

function trayIcon(active) {
  // 16x16 template circle; filled while dictating
  const size = 16;
  const canvas = Buffer.alloc(size * size * 4);
  const cx = 7.5, cy = 7.5, rOuter = 6.5, rInner = active ? 6.5 : 4.5;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const d = Math.hypot(x - cx, y - cy);
      const on = active ? d <= rOuter : (d <= rOuter && d >= rInner - 1.2);
      const i = (y * size + x) * 4;
      canvas[i] = 0; canvas[i + 1] = 0; canvas[i + 2] = 0;
      canvas[i + 3] = on ? 255 : 0;
    }
  }
  const image = nativeImage.createFromBuffer(canvas, { width: size, height: size });
  image.setTemplateImage(true);
  return image;
}

function openSettings() {
  if (settingsWindow) { settingsWindow.focus(); return; }
  settingsWindow = new BrowserWindow({
    width: 900,
    height: 680,
    title: 'WisperFree Settings',
    webPreferences: { contextIsolation: true },
  });
  settingsWindow.loadFile(path.join(__dirname, 'settings.html'));
  settingsWindow.on('closed', () => { settingsWindow = null; });
}

async function refreshMenu() {
  try {
    const status = await api('/api/status');
    dictating = status.dictating;
  } catch { /* daemon offline */ }
  tray.setImage(trayIcon(dictating));
  tray.setToolTip(dictating ? 'WisperFree — listening' : 'WisperFree');
  tray.setContextMenu(Menu.buildFromTemplate([
    {
      label: dictating ? 'Stop dictation' : 'Start dictation',
      click: async () => {
        try {
          const result = await api('/api/dictation/toggle', { method: 'POST' });
          dictating = result.dictating;
        } catch (err) { console.error(err); }
        refreshMenu();
      },
    },
    { type: 'separator' },
    { label: 'Settings…', click: openSettings },
    {
      label: 'Ollama site (model downloads)',
      click: () => shell.openExternal('https://ollama.com'),
    },
    { type: 'separator' },
    { label: 'Quit WisperFree', click: () => app.quit() },
  ]));
}

app.whenReady().then(async () => {
  if (app.dock) app.dock.hide(); // menu-bar-only on macOS
  tray = new Tray(trayIcon(false));
  await ensureDaemon();
  await refreshMenu();
  setInterval(refreshMenu, 3000);
});

app.on('window-all-closed', (event) => {
  // Tray app: stay alive with no windows
});

app.on('quit', () => {
  if (daemonProcess) daemonProcess.kill();
});
