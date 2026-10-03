const $ = (selector) => document.querySelector(selector);

function formatTime(value) {
  return value ? new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'medium' }) : 'Unknown time';
}

function imageUrl(path) {
  if (!path) return '';
  const normalized = path.replaceAll('\\', '/');
  const marker = normalized.indexOf('logs/');
  return marker >= 0 ? `/media/${normalized.slice(marker)}` : '';
}

function renderEvents(events) {
  const target = $('#events');
  if (!events.length) { target.innerHTML = '<div class="empty">No movement events recorded yet.</div>'; return; }
  target.innerHTML = events.map((event) => `
    <div class="event-row">
      ${imageUrl(event.image_path) ? `<img class="thumb" src="${imageUrl(event.image_path)}" alt="Face crop">` : '<div class="thumb"></div>'}
      <div><div class="event-title">${event.name || event.face_id || 'Unknown face'}</div><div class="event-meta">DOB ${event.date_of_birth || '-'} · Track ${event.track_id ?? '-'} · ${event.event_type.toUpperCase()} TIME ${formatTime(event.timestamp)}</div></div>
      <span class="badge ${event.event_type}">${event.event_type}</span>
    </div>`).join('');
}

function renderAudit(audit) {
  const target = $('#audit');
  if (!audit.length) { target.innerHTML = '<div class="empty">No audit activity yet.</div>'; return; }
  target.innerHTML = audit.map((event) => `<div class="audit-row"><div class="audit-type">${event.event_type}</div><div class="audit-payload">${JSON.stringify(event.payload || {})}</div><div class="audit-time">${formatTime(event.timestamp)}</div></div>`).join('');
}

async function refresh() {
  try {
    const response = await fetch('/api/overview', { cache: 'no-store' });
    const data = await response.json();
    const connected = data.ok;
    $('.pulse').style.background = connected ? '#82b84d' : '#db9d42';
    $('#connection-label').textContent = connected ? 'MongoDB connected' : 'MongoDB unavailable';
    if (!connected) return;
    $('#today-total').textContent = data.stats.today_total;
    $('#today-entries').textContent = data.stats.today_entries;
    $('#today-exits').textContent = data.stats.today_exits;
    $('#known-faces').textContent = data.stats.known_faces;
    const identity = data.identities?.[0];
    $('#identity-card').innerHTML = identity ? `<span class="label">SAVED IDENTITY</span><strong>${identity.name || identity.face_id}</strong><code>DOB ${identity.date_of_birth || '-'}</code><small>Registered ${formatTime(identity.registered_at)}</small>` : '<span class="label">SAVED IDENTITY</span><small>No identity enrolled</small>';
    renderEvents(data.events);
    renderAudit(data.audit);
  } catch (error) {
    $('#connection-label').textContent = 'Dashboard offline';
  }
}

refresh();
setInterval(refresh, 5000);
