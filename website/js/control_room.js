/**
 * SafeKid AI — Security Control Room Interactive Controller
 *
 * Handles:
 * - Real-time system health and hardware diagnostics polling
 * - Multi-video drag-and-drop ingestion & demo video loading
 * - Video processing queue management & progress tracking
 * - Video playback and annotated detection stream switching
 * - Interactive CCTV Hazard Geofence Editor (canvas polygon drawing & persistence)
 * - Real-time Incident Action Queue (human verification, acknowledgement, resolution)
 * - Community Central Messaging Gateway configuration & test alert dispatch
 * - ESP32 Physical Buzzer hardware configuration, status polling, and actuation controls
 * - Filterable historical incident log with CSV/JSON export
 */

class SafeKidControlRoom {
  constructor() {
    this.jobs = [];
    this.activeJobId = null;
    this.activeJob = null;
    this.roiConfig = null;
    this.incidents = [];
    this.roiCanvas = null;
    this.roiCtx = null;
    this.roiFrameImage = new Image();
    this.currentPolygonPoints = [];
    this.pollingTimer = null;
    this.pollIntervalMs = 2000;
  }

  init() {
    console.log('[SafeKid Control Room] Initializing operator interface...');
    this.setupDOMReferences();
    this.setupEventListeners();
    this.setupDropzone();
    this.setupROICanvas();

    // Initial data load
    this.refreshDashboard();

    // Start background status polling
    this.startPolling();
  }

  setupDOMReferences() {
    // Status badges
    this.elStatusPipeline = document.getElementById('status-pipeline');
    this.elStatusESP32 = document.getElementById('status-esp32');
    this.elLabelESP32 = document.getElementById('label-esp32-status');
    this.elStatusNotif = document.getElementById('status-notif');
    this.elLabelNotif = document.getElementById('label-notif-status');
    this.elStatActiveIncidents = document.getElementById('stat-active-incidents');
    this.elStatCompletedJobs = document.getElementById('stat-completed-jobs');

    // Video workspace
    this.dropzone = document.getElementById('upload-dropzone');
    this.fileInput = document.getElementById('video-file-input');
    this.jobQueueWrapper = document.getElementById('job-queue-wrapper');
    this.tbodyJobQueue = document.getElementById('tbody-job-queue');
    this.selectActiveJob = document.getElementById('select-active-job');
    this.mainVideoPlayer = document.getElementById('main-video-player');
    this.diagVideoName = document.getElementById('diag-video-name');
    this.diagResolution = document.getElementById('diag-resolution');
    this.diagFps = document.getElementById('diag-fps');
    this.diagTotalFrames = document.getElementById('diag-total-frames');
    this.diagProcessedFrames = document.getElementById('diag-processed-frames');
    this.diagProcFps = document.getElementById('diag-proc-fps');
    this.diagActiveTracks = document.getElementById('diag-active-tracks');
    this.diagIncidentsFound = document.getElementById('diag-incidents-found');
    this.activeProcIndicator = document.getElementById('active-processing-indicator');
    this.liveProgressBar = document.getElementById('live-progress-bar');

    // ROI Editor
    this.roiCanvas = document.getElementById('roi-editor-canvas');
    if (this.roiCanvas) {
      this.roiCtx = this.roiCanvas.getContext('2d');
    }
    this.lblPolygonPoints = document.getElementById('lbl-polygon-points');
    this.listActiveZones = document.getElementById('list-active-zones');
    this.lblZoneCount = document.getElementById('lbl-zone-count');

    // Incidents
    this.incidentCardsContainer = document.getElementById('incident-cards-container');
    this.badgeIncidentFeedCount = document.getElementById('badge-incident-feed-count');
    this.tbodyIncidentHistory = document.getElementById('tbody-incident-history');

    // Notification Panel
    this.selectNotifProvider = document.getElementById('select-notif-provider');
    this.fieldWebhookUrl = document.getElementById('field-webhook-url');
    this.inputWebhookUrl = document.getElementById('input-webhook-url');
    this.fieldsTelegram = document.getElementById('fields-telegram');
    this.inputTelegramToken = document.getElementById('input-telegram-token');
    this.inputTelegramChat = document.getElementById('input-telegram-chat');
    this.badgeMessagingStatus = document.getElementById('badge-messaging-status');
    this.notifDispatchLog = document.getElementById('notif-dispatch-log');

    // ESP32 Panel
    this.badgeESP32Status = document.getElementById('badge-esp32-status');
    this.inputESP32Host = document.getElementById('input-esp32-host');
    this.inputESP32Port = document.getElementById('input-esp32-port');
    this.inputESP32Cooldown = document.getElementById('input-esp32-cooldown');
    this.inputESP32Duration = document.getElementById('input-esp32-duration');
    this.inputESP32Token = document.getElementById('input-esp32-token');
    this.esp32ActuationLog = document.getElementById('esp32-actuation-log');
  }

  setupEventListeners() {
    // Refresh button
    const btnRefresh = document.getElementById('btn-refresh-dashboard');
    if (btnRefresh) {
      btnRefresh.addEventListener('click', () => this.refreshDashboard());
    }

    // Demo video buttons
    document.querySelectorAll('.btn-demo-pill').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const demoKey = e.currentTarget.getAttribute('data-demo');
        this.loadDemoVideo(demoKey);
      });
    });

    // Active job select
    if (this.selectActiveJob) {
      this.selectActiveJob.addEventListener('change', (e) => {
        const jid = e.target.value;
        this.setActiveJob(jid);
      });
    }

    // Capture Frame Snapshot button
    const btnCaptureFrame = document.getElementById('btn-reload-roi-frame');
    if (btnCaptureFrame) {
      btnCaptureFrame.addEventListener('click', () => this.loadROIFrame(this.activeJobId));
    }

    // ROI undo / clear buttons
    const btnUndo = document.getElementById('btn-roi-undo-point');
    if (btnUndo) {
      btnUndo.addEventListener('click', () => {
        if (this.currentPolygonPoints.length > 0) {
          this.currentPolygonPoints.pop();
          this.renderROICanvas();
        }
      });
    }

    const btnClear = document.getElementById('btn-roi-clear-points');
    if (btnClear) {
      btnClear.addEventListener('click', () => {
        this.currentPolygonPoints = [];
        this.renderROICanvas();
      });
    }

    // Save New Zone button
    const btnSaveZone = document.getElementById('btn-save-new-zone');
    if (btnSaveZone) {
      btnSaveZone.addEventListener('click', () => this.saveNewZoneGeofence());
    }

    // Notification provider change
    if (this.selectNotifProvider) {
      this.selectNotifProvider.addEventListener('change', (e) => {
        this.updateNotifFieldsVisibility(e.target.value);
      });
    }

    // Notification save & test buttons
    const btnSaveNotif = document.getElementById('btn-save-notif-config');
    if (btnSaveNotif) {
      btnSaveNotif.addEventListener('click', () => this.saveNotificationConfig());
    }
    const btnTestNotif = document.getElementById('btn-test-notif-dispatch');
    if (btnTestNotif) {
      btnTestNotif.addEventListener('click', () => this.sendTestNotification());
    }

    // ESP32 save, test, and silence buttons
    const btnSaveESP32 = document.getElementById('btn-save-esp32-config');
    if (btnSaveESP32) {
      btnSaveESP32.addEventListener('click', () => this.saveESP32Config());
    }
    const btnPulse = document.getElementById('btn-esp32-test-pulse');
    if (btnPulse) {
      btnPulse.addEventListener('click', () => this.triggerESP32Pulse());
    }
    const btnSilence = document.getElementById('btn-esp32-silence');
    if (btnSilence) {
      btnSilence.addEventListener('click', () => this.silenceESP32());
    }

    // Incident search & filters
    ['filter-incident-search', 'filter-incident-severity', 'filter-incident-status'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('input', () => this.renderIncidentHistoryTable());
        el.addEventListener('change', () => this.renderIncidentHistoryTable());
      }
    });

    // Incident Export buttons
    const btnExpCsv = document.getElementById('btn-export-incidents-csv');
    if (btnExpCsv) {
      btnExpCsv.addEventListener('click', () => this.exportIncidentsCSV());
    }
    const btnExpJson = document.getElementById('btn-export-incidents-json');
    if (btnExpJson) {
      btnExpJson.addEventListener('click', () => this.exportIncidentsJSON());
    }
  }

  // =========================================================================
  // DRAG & DROP UPLOAD
  // =========================================================================

  setupDropzone() {
    if (!this.dropzone || !this.fileInput) return;

    ['dragenter', 'dragover'].forEach(name => {
      this.dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      this.dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.dropzone.classList.remove('dragover');
      });
    });

    this.dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        this.uploadVideoFiles(files);
      }
    });

    this.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        this.uploadVideoFiles(e.target.files);
      }
    });
  }

  async uploadVideoFiles(fileList) {
    const formData = new FormData();
    let validCount = 0;

    for (let i = 0; i < fileList.length; i++) {
      const f = fileList[i];
      const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase();
      if (['.mp4', '.avi', '.mov', '.mkv'].includes(ext)) {
        formData.append('video', f);
        validCount++;
      } else {
        alert(`File "${f.name}" rejected: format must be MP4, AVI, or MOV.`);
      }
    }

    if (validCount === 0) return;

    this.showGlobalNotice(`Uploading ${validCount} video(s)...`, 'info');

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.status === 'success') {
        this.showGlobalNotice(`Successfully uploaded ${data.data.uploaded_count} video(s).`, 'success');
        await this.loadJobs();
        if (data.data.jobs && data.data.jobs.length > 0) {
          this.setActiveJob(data.data.jobs[0].job_id);
        }
      } else {
        alert(`Upload error: ${data.message}`);
      }
    } catch (err) {
      console.error('Upload failed:', err);
      alert(`Network error during video upload: ${err.message}`);
    }
  }

  async loadDemoVideo(demoKey) {
    try {
      const res = await fetch('/api/jobs/demo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ demo_name: demoKey }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        this.showGlobalNotice(`Loaded demo recording fixture: ${data.data.job.filename}`, 'success');
        await this.loadJobs();
        this.setActiveJob(data.data.job.job_id);
      } else {
        alert(`Could not load demo video: ${data.message}`);
      }
    } catch (err) {
      console.error('Error loading demo video:', err);
    }
  }

  // =========================================================================
  // JOB MANAGEMENT & QUEUE
  // =========================================================================

  async loadJobs() {
    try {
      const res = await fetch('/api/jobs');
      const data = await res.json();
      if (data.status === 'success') {
        this.jobs = data.data || [];
        this.renderJobQueue();
        this.updateActiveJobSelect();
      }
    } catch (err) {
      console.warn('Could not fetch jobs list:', err);
    }
  }

  renderJobQueue() {
    if (!this.tbodyJobQueue || !this.jobQueueWrapper) return;

    if (this.jobs.length === 0) {
      this.jobQueueWrapper.style.display = 'none';
      return;
    }

    this.jobQueueWrapper.style.display = 'block';
    this.tbodyJobQueue.innerHTML = this.jobs.map(job => {
      let statusBadge = '<span class="card-badge badge-baseline">QUEUED</span>';
      if (job.status === 'PROCESSING') {
        statusBadge = '<span class="card-badge badge-warning" style="animation: pulse 1.5s infinite;">PROCESSING</span>';
      } else if (job.status === 'COMPLETED') {
        statusBadge = '<span class="card-badge badge-implemented">COMPLETED</span>';
      } else if (job.status === 'FAILED') {
        statusBadge = '<span class="card-badge badge-planned">FAILED</span>';
      } else if (job.status === 'CANCELLED') {
        statusBadge = '<span class="card-badge badge-evaluated">CANCELLED</span>';
      }

      const pct = job.progress_percent || 0.0;
      const isProcessing = job.status === 'PROCESSING';
      const isQueued = job.status === 'QUEUED';

      return `
        <tr data-job-id="${job.job_id}">
          <td>
            <strong>${job.filename}</strong>
            <div style="font-size:0.72rem; color:var(--text-muted); font-family:var(--font-mono);">${job.job_id}</div>
          </td>
          <td>${job.file_size_mb || 0} MB</td>
          <td>${job.total_frames || 0} f (${job.fps || 25} fps)</td>
          <td>${statusBadge}</td>
          <td style="min-width: 140px;">
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:var(--text-secondary);">
              <span>${pct}%</span>
              <span>${job.current_frame || 0}/${job.total_frames || 0}</span>
            </div>
            <div class="job-progress-bar-wrap">
              <div class="job-progress-bar-fill" style="width: ${pct}%;"></div>
            </div>
          </td>
          <td>
            <strong style="color:${job.incident_count > 0 ? 'var(--rose-primary)' : 'var(--text-muted)'};">
              ${job.incident_count || 0}
            </strong>
          </td>
          <td>
            <div style="display:flex; gap:0.35rem;">
              ${(isQueued || job.status === 'FAILED') ? `
                <button class="btn btn-primary" style="font-size:0.72rem; padding:0.25rem 0.5rem;" onclick="window.safekidCR.startJobProcess('${job.job_id}')">
                  ▶ Process
                </button>
              ` : ''}
              ${isProcessing ? `
                <button class="btn btn-secondary" style="font-size:0.72rem; padding:0.25rem 0.5rem;" onclick="window.safekidCR.cancelJob('${job.job_id}')">
                  ⏹ Cancel
                </button>
              ` : ''}
              <button class="btn btn-secondary" style="font-size:0.72rem; padding:0.25rem 0.4rem; color:var(--rose-primary);" onclick="window.safekidCR.deleteJob('${job.job_id}')" title="Delete job">
                🗑
              </button>
            </div>
          </td>
        </tr>
      `;
    }).join('');
  }

  updateActiveJobSelect() {
    if (!this.selectActiveJob) return;
    const currentVal = this.selectActiveJob.value;

    this.selectActiveJob.innerHTML = '<option value="">-- Select Video --</option>' + this.jobs.map(j => {
      const tag = j.status === 'COMPLETED' ? '✓ ' : '';
      return `<option value="${j.job_id}">${tag}${j.filename} (${j.status})</option>`;
    }).join('');

    if (this.activeJobId && this.jobs.some(j => j.job_id === this.activeJobId)) {
      this.selectActiveJob.value = this.activeJobId;
    } else if (currentVal && this.jobs.some(j => j.job_id === currentVal)) {
      this.selectActiveJob.value = currentVal;
    } else if (this.jobs.length > 0) {
      this.setActiveJob(this.jobs[0].job_id);
    }
  }

  async setActiveJob(jobId) {
    this.activeJobId = jobId;
    if (this.selectActiveJob) {
      this.selectActiveJob.value = jobId || '';
    }

    this.activeJob = this.jobs.find(j => j.job_id === jobId) || null;
    this.updateVideoWorkspace();
    this.loadROIFrame(jobId);
  }

  updateVideoWorkspace() {
    if (!this.activeJob) {
      if (this.diagVideoName) this.diagVideoName.textContent = 'None';
      return;
    }

    const j = this.activeJob;
    if (this.diagVideoName) this.diagVideoName.textContent = j.filename;
    if (this.diagResolution) this.diagResolution.textContent = `${j.frame_width || 848} x ${j.frame_height || 480}`;
    if (this.diagFps) this.diagFps.textContent = `${j.fps || 25} fps`;
    if (this.diagTotalFrames) this.diagTotalFrames.textContent = j.total_frames || 0;
    if (this.diagProcessedFrames) this.diagProcessedFrames.textContent = j.current_frame || 0;
    if (this.diagProcFps) this.diagProcFps.textContent = `${j.processing_fps || 0.0} fps`;
    if (this.diagIncidentsFound) this.diagIncidentsFound.textContent = j.incident_count || 0;

    // Show/hide live processing indicator
    if (this.activeProcIndicator && this.liveProgressBar) {
      if (j.status === 'PROCESSING') {
        this.activeProcIndicator.style.display = 'block';
        this.liveProgressBar.style.width = `${j.progress_percent || 0}%`;
      } else {
        this.activeProcIndicator.style.display = 'none';
      }
    }

    // Set video src in player
    if (this.mainVideoPlayer) {
      let src = '';
      if (j.annotated_video_url && j.status === 'COMPLETED') {
        src = j.annotated_video_url;
      } else if (j.file_path) {
        // Stream original uploaded file
        src = `/media/uploads/${j.file_path.split(/[\\/]/).pop()}`;
      }
      if (src && !this.mainVideoPlayer.src.endsWith(src)) {
        this.mainVideoPlayer.src = src;
        this.mainVideoPlayer.load();
      }
    }
  }

  async startJobProcess(jobId) {
    try {
      const res = await fetch(`/api/jobs/${jobId}/process`, { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        this.showGlobalNotice(`Vision pipeline started for ${jobId}`, 'info');
        await this.loadJobs();
      } else {
        alert(`Could not start processing: ${data.message}`);
      }
    } catch (err) {
      console.error('Error starting job:', err);
    }
  }

  async cancelJob(jobId) {
    try {
      await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
      await this.loadJobs();
    } catch (err) {
      console.error('Error cancelling job:', err);
    }
  }

  async deleteJob(jobId) {
    if (!confirm(`Delete video recording and processing results for ${jobId}?`)) return;
    try {
      await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
      if (this.activeJobId === jobId) {
        this.activeJobId = null;
        this.activeJob = null;
      }
      await this.loadJobs();
      await this.loadIncidents();
    } catch (err) {
      console.error('Error deleting job:', err);
    }
  }

  // =========================================================================
  // INTERACTIVE HAZARD ZONE GEOFENCE EDITOR (ROI CANVAS)
  // =========================================================================

  setupROICanvas() {
    if (!this.roiCanvas) return;

    this.roiCanvas.addEventListener('click', (e) => {
      const rect = this.roiCanvas.getBoundingClientRect();
      const scaleX = this.roiCanvas.width / rect.width;
      const scaleY = this.roiCanvas.height / rect.height;

      const clickX = Math.round((e.clientX - rect.left) * scaleX);
      const clickY = Math.round((e.clientY - rect.top) * scaleY);

      this.currentPolygonPoints.push([clickX, clickY]);
      this.renderROICanvas();
    });

    this.roiFrameImage.onload = () => {
      this.renderROICanvas();
    };

    // Load initial reference frame
    this.loadROIFrame(null);
  }

  async loadROIFrame(jobId) {
    try {
      const url = jobId ? `/api/roi/frame?job_id=${jobId}` : '/api/roi/frame';
      const res = await fetch(url);
      const data = await res.json();
      if (data.status === 'success' && data.data.image_data) {
        this.roiFrameImage.src = data.data.image_data;
        if (data.data.width && data.data.height && this.roiCanvas) {
          this.roiCanvas.width = data.data.width;
          this.roiCanvas.height = data.data.height;
        }
      }
    } catch (err) {
      console.warn('Could not load ROI reference frame:', err);
    }
  }

  async loadROIConfig() {
    try {
      const res = await fetch('/api/roi');
      const data = await res.json();
      if (data.status === 'success') {
        this.roiConfig = data.data;
        this.renderROICanvas();
        this.renderActiveZonesList();
      }
    } catch (err) {
      console.warn('Could not load ROI configuration:', err);
    }
  }

  renderROICanvas() {
    if (!this.roiCanvas || !this.roiCtx) return;
    const ctx = this.roiCtx;
    const w = this.roiCanvas.width;
    const h = this.roiCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // 1. Draw video reference frame image if loaded
    if (this.roiFrameImage.complete && this.roiFrameImage.naturalWidth > 0) {
      ctx.drawImage(this.roiFrameImage, 0, 0, w, h);
    } else {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, w, h);
    }

    // 2. Draw existing configured zones
    if (this.roiConfig && this.roiConfig.zones) {
      this.roiConfig.zones.forEach((z, idx) => {
        if (!z.enabled) return;
        const pts = z.polygon || [];
        if (pts.length < 3) return;

        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i][0], pts[i][1]);
        }
        ctx.closePath();

        const isDanger = (z.policy && z.policy.zone_role === 'DANGER') || z.severity === 'CRITICAL';
        ctx.fillStyle = isDanger ? 'rgba(244, 63, 94, 0.22)' : 'rgba(245, 158, 11, 0.20)';
        ctx.fill();

        ctx.strokeStyle = isDanger ? '#f43f5e' : '#f59e0b';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // Draw zone name label
        const cx = pts.reduce((sum, p) => sum + p[0], 0) / pts.length;
        const cy = pts.reduce((sum, p) => sum + p[1], 0) / pts.length;

        ctx.fillStyle = 'rgba(7, 11, 20, 0.85)';
        ctx.font = 'bold 12px Inter, sans-serif';
        const txt = `[${z.severity}] ${z.name}`;
        const tm = ctx.measureText(txt);
        ctx.fillRect(cx - tm.width / 2 - 4, cy - 10, tm.width + 8, 18);

        ctx.fillStyle = isDanger ? '#fb7185' : '#fcd34d';
        ctx.fillText(txt, cx - tm.width / 2, cy + 3);
      });
    }

    // 3. Draw in-progress polygon
    const pts = this.currentPolygonPoints;
    if (this.lblPolygonPoints) {
      this.lblPolygonPoints.textContent = pts.length;
    }

    if (pts.length > 0) {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i][0], pts[i][1]);
      }
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw vertex dots
      pts.forEach((p, idx) => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
        ctx.fillStyle = idx === 0 ? '#10b981' : '#06b6d4';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    }
  }

  renderActiveZonesList() {
    if (!this.listActiveZones) return;
    const zones = (this.roiConfig && this.roiConfig.zones) || [];
    if (this.lblZoneCount) this.lblZoneCount.textContent = zones.length;

    if (zones.length === 0) {
      this.listActiveZones.innerHTML = '<div style="color:var(--text-muted); font-size:0.75rem;">No zones configured.</div>';
      return;
    }

    this.listActiveZones.innerHTML = zones.map((z, idx) => {
      const isCrit = z.severity === 'CRITICAL';
      return `
        <div class="zone-chip-item ${isCrit ? 'critical' : 'high'}">
          <div>
            <strong style="font-size:0.8rem; color:#fff;">${z.name}</strong>
            <div style="font-size:0.7rem; color:var(--text-secondary);">
              ${z.zone_type} | Role: ${z.policy ? z.policy.zone_role : 'DANGER'} | Dwell: ${z.dwell_time_seconds || 0}s
            </div>
          </div>
          <button class="btn btn-secondary" style="font-size:0.7rem; padding:0.2rem 0.4rem; color:var(--rose-primary);" onclick="window.safekidCR.deleteZone(${idx})">
            ✕
          </button>
        </div>
      `;
    }).join('');
  }

  async saveNewZoneGeofence() {
    if (this.currentPolygonPoints.length < 3) {
      alert('Please click on the video frame to place at least 3 points to enclose a polygon boundary.');
      return;
    }

    const inputName = document.getElementById('input-zone-name');
    const selectType = document.getElementById('select-zone-type');
    const selectRole = document.getElementById('select-zone-role');
    const selectSeverity = document.getElementById('select-zone-severity');
    const inputDwell = document.getElementById('input-zone-dwell');

    const zoneName = (inputName && inputName.value.trim()) || 'Configured Hazard Zone';
    const zoneType = (selectType && selectType.value) || 'SWIMMING_POOL_BOUNDARY';
    const zoneRole = (selectRole && selectRole.value) || 'DANGER';
    const severity = (selectSeverity && selectSeverity.value) || 'CRITICAL';
    const dwell = parseFloat(inputDwell ? inputDwell.value : '2.0') || 2.0;

    const newZone = {
      roi_id: `zone_${Date.now()}`,
      name: zoneName,
      zone_type: zoneType,
      severity: severity,
      enabled: true,
      dwell_time_seconds: dwell,
      polygon: this.currentPolygonPoints.map(p => [p[0], p[1]]),
      policy: {
        zone_role: zoneRole,
        entry_alert: zoneRole === 'DANGER',
        exit_alert: false,
        dwell_alert: dwell > 0,
        dwell_threshold_seconds: dwell,
        severity: severity,
      },
    };

    const updatedConfig = {
      community_id: (this.roiConfig && this.roiConfig.community_id) || 'palm_grove_residential',
      camera_id: (this.roiConfig && this.roiConfig.camera_id) || 'cam_main_pool',
      frame_width: (this.roiCanvas && this.roiCanvas.width) || 848,
      frame_height: (this.roiCanvas && this.roiCanvas.height) || 480,
      version: '1.0',
      zones: [...((this.roiConfig && this.roiConfig.zones) || []), newZone],
    };

    try {
      const res = await fetch('/api/roi', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedConfig),
      });
      const data = await res.json();
      if (data.status === 'success') {
        this.roiConfig = data.data;
        this.currentPolygonPoints = [];
        this.renderROICanvas();
        this.renderActiveZonesList();
        this.showGlobalNotice(`Saved geofence zone: "${zoneName}"`, 'success');
      } else {
        alert(`Error saving ROI zone: ${data.message}`);
      }
    } catch (err) {
      console.error('Error saving ROI:', err);
    }
  }

  async deleteZone(idx) {
    if (!this.roiConfig || !this.roiConfig.zones) return;
    const removed = this.roiConfig.zones.splice(idx, 1);
    try {
      const res = await fetch('/api/roi', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(this.roiConfig),
      });
      const data = await res.json();
      if (data.status === 'success') {
        this.roiConfig = data.data;
        this.renderROICanvas();
        this.renderActiveZonesList();
      }
    } catch (err) {
      console.error('Error removing zone:', err);
    }
  }

  // =========================================================================
  // REAL-TIME INCIDENT FEED & VERIFICATION QUEUE
  // =========================================================================

  async loadIncidents() {
    try {
      const res = await fetch('/api/incidents');
      const data = await res.json();
      if (data.status === 'success') {
        this.incidents = data.data || [];
        this.renderIncidentCards();
        this.renderIncidentHistoryTable();
      }
    } catch (err) {
      console.warn('Could not load incidents:', err);
    }
  }

  renderIncidentCards() {
    if (!this.incidentCardsContainer) return;

    // Filter to active or acknowledged incidents for the live feed
    const openIncidents = this.incidents.filter(i => ['ACTIVE', 'ACKNOWLEDGED'].includes(i.state));

    if (this.badgeIncidentFeedCount) {
      this.badgeIncidentFeedCount.textContent = `${openIncidents.length} Open Incident(s)`;
    }
    if (this.elStatActiveIncidents) {
      this.elStatActiveIncidents.textContent = openIncidents.length;
    }

    if (openIncidents.length === 0) {
      this.incidentCardsContainer.innerHTML = `
        <div style="grid-column: 1 / -1; text-align:center; padding:2.5rem; color:var(--text-muted); background:var(--bg-secondary); border-radius:var(--radius-md);">
          <div style="font-size:1.8rem; margin-bottom:0.5rem;">🛡️</div>
          <strong>No active safety incidents detected yet.</strong>
          <div style="font-size:0.82rem; margin-top:0.25rem;">Upload and process a video recording above to evaluate hazard geofences.</div>
        </div>
      `;
      return;
    }

    this.incidentCardsContainer.innerHTML = openIncidents.map(inc => {
      const isCrit = inc.severity === 'CRITICAL';
      const isAck = inc.state === 'ACKNOWLEDGED';
      const cardClass = isAck ? 'acknowledged' : (isCrit ? 'critical' : 'high');

      return `
        <div class="incident-card ${cardClass}">
          <div class="incident-card-top">
            <div>
              <span class="card-badge ${isCrit ? 'badge-planned' : 'badge-warning'}">${inc.severity} HAZARD</span>
              <h4 style="font-size:0.95rem; font-weight:700; color:#fff; margin-top:0.35rem;">${inc.primary_zone_name}</h4>
            </div>
            <span class="status-pill ${isAck ? 'online' : 'disconnected'}" style="font-size:0.7rem;">
              ${inc.state}
            </span>
          </div>

          ${inc.thumbnail_url ? `
            <img src="${inc.thumbnail_url}" alt="Incident Snapshot" class="incident-thumb-img" onerror="this.style.display='none';">
          ` : ''}

          <table class="incident-meta-table">
            <tr><td>Incident ID</td><td><code>${inc.incident_id}</code></td></tr>
            <tr><td>Classification</td><td><strong>${inc.classification}</strong> (Track #${inc.track_id})</td></tr>
            <tr><td>Source Recording</td><td>${inc.source_video || 'CCTV Feed'}</td></tr>
            <tr><td>Detection Time</td><td>${inc.timestamp_str || 'N/A'}</td></tr>
            <tr><td>Notification</td><td><span class="card-badge ${inc.notification_status === 'DELIVERED' ? 'badge-implemented' : 'badge-planned'}">${inc.notification_status || 'PENDING'}</span></td></tr>
            <tr><td>ESP32 Buzzer</td><td><span class="card-badge ${inc.buzzer_status === 'TRIGGERED' ? 'badge-implemented' : 'badge-warning'}">${inc.buzzer_status || 'NOT_CONNECTED'}</span></td></tr>
          </table>

          <div class="incident-actions-row">
            ${!isAck ? `
              <button class="btn-action-ack" onclick="window.safekidCR.acknowledgeIncident('${inc.incident_id}')">
                ✓ Acknowledge
              </button>
            ` : ''}
            <button class="btn-action-resolve" onclick="window.safekidCR.resolveIncident('${inc.incident_id}')">
              🔒 Mark Resolved
            </button>
          </div>
        </div>
      `;
    }).join('');
  }

  async acknowledgeIncident(incidentId) {
    const operator = prompt('Enter operator name or callsign:', 'Control Room Guard') || 'Operator';
    try {
      const res = await fetch(`/api/incidents/${incidentId}/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ operator_name: operator, notes: 'Acknowledged via Control Room dashboard' }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        this.showGlobalNotice(`Incident ${incidentId} marked ACKNOWLEDGED.`, 'info');
        await this.loadIncidents();
      }
    } catch (err) {
      console.error('Error acknowledging incident:', err);
    }
  }

  async resolveIncident(incidentId) {
    const reason = prompt('Enter resolution notes / action taken:', 'Child escorted away by guardian') || 'Resolved by operator';
    try {
      const res = await fetch(`/api/incidents/${incidentId}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ operator_name: 'Security Guard', reason: reason }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        this.showGlobalNotice(`Incident ${incidentId} marked RESOLVED.`, 'success');
        await this.loadIncidents();
      }
    } catch (err) {
      console.error('Error resolving incident:', err);
    }
  }

  // =========================================================================
  // NOTIFICATIONS GATEWAY
  // =========================================================================

  async loadNotificationConfig() {
    try {
      const res = await fetch('/api/notifications/config');
      const data = await res.json();
      if (data.status === 'success') {
        const s = data.data;
        if (this.selectNotifProvider) this.selectNotifProvider.value = s.provider_type || 'NONE';
        this.updateNotifFieldsVisibility(s.provider_type || 'NONE');

        if (this.badgeMessagingStatus) {
          this.badgeMessagingStatus.textContent = s.status;
          this.badgeMessagingStatus.className = s.status === 'CONFIGURED' ? 'card-badge badge-implemented' : 'card-badge badge-planned';
        }
        if (this.elLabelNotif) {
          this.elLabelNotif.textContent = `Community Alerts: ${s.status}`;
        }
        if (this.elStatusNotif) {
          this.elStatusNotif.className = `status-pill ${s.status === 'CONFIGURED' ? 'online' : 'unconfigured'}`;
        }
      }
    } catch (err) {
      console.warn('Could not load notification config:', err);
    }

    // Load recent history
    try {
      const resHist = await fetch('/api/notifications/history');
      const dataHist = await resHist.json();
      if (dataHist.status === 'success' && this.notifDispatchLog) {
        const list = dataHist.data || [];
        if (list.length === 0) {
          this.notifDispatchLog.textContent = 'No messages dispatched yet.';
        } else {
          this.notifDispatchLog.innerHTML = list.slice(0, 5).map(e => `
            <div>[${e.timestamp || ''}] <strong>${e.provider || ''}</strong>: ${e.message} (${e.success ? '✓' : '✕'})</div>
          `).join('');
        }
      }
    } catch (err) {}
  }

  updateNotifFieldsVisibility(providerType) {
    if (this.fieldWebhookUrl) {
      this.fieldWebhookUrl.style.display = providerType === 'WEBHOOK' ? 'flex' : 'none';
    }
    if (this.fieldsTelegram) {
      this.fieldsTelegram.style.display = providerType === 'TELEGRAM' ? 'block' : 'none';
    }
  }

  async saveNotificationConfig() {
    const ptype = this.selectNotifProvider.value;
    const webhook = this.inputWebhookUrl ? this.inputWebhookUrl.value.trim() : '';
    const tgToken = this.inputTelegramToken ? this.inputTelegramToken.value.trim() : '';
    const tgChat = this.inputTelegramChat ? this.inputTelegramChat.value.trim() : '';

    try {
      const res = await fetch('/api/notifications/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider_type: ptype,
          webhook_url: webhook,
          telegram_token: tgToken,
          telegram_chat_id: tgChat,
        }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        this.showGlobalNotice('Notification configuration saved.', 'success');
        this.loadNotificationConfig();
      }
    } catch (err) {
      console.error('Error saving notification config:', err);
    }
  }

  async sendTestNotification() {
    try {
      const res = await fetch('/api/notifications/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: 'Diagnostic alert pulse from SafeKid AI Control Room' }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        const result = data.data;
        if (result.success) {
          this.showGlobalNotice(`Alert delivered via ${result.provider}: ${result.message}`, 'success');
        } else {
          alert(`Notification dispatch failed (${result.provider}):\n${result.message}`);
        }
        this.loadNotificationConfig();
      }
    } catch (err) {
      console.error('Error testing notification:', err);
    }
  }

  // =========================================================================
  // ESP32 PHYSICAL BUZZER HARDWARE
  // =========================================================================

  async loadESP32Status() {
    try {
      const res = await fetch('/api/esp32/status');
      const data = await res.json();
      if (data.status === 'success') {
        const s = data.data;
        const health = s.health || {};
        const cfg = s.config || {};

        if (this.inputESP32Host) this.inputESP32Host.value = cfg.host || '192.168.1.120';
        if (this.inputESP32Port) this.inputESP32Port.value = cfg.port || 80;
        if (this.inputESP32Cooldown) this.inputESP32Cooldown.value = cfg.cooldown_seconds || 10;
        if (this.inputESP32Duration) this.inputESP32Duration.value = cfg.default_duration_ms || 1500;

        const isConn = health.connected === true;
        const statusLabel = health.status || 'NOT CONNECTED';

        if (this.badgeESP32Status) {
          this.badgeESP32Status.textContent = statusLabel;
          this.badgeESP32Status.className = isConn ? 'card-badge badge-implemented' : 'card-badge badge-warning';
        }
        if (this.elLabelESP32) {
          this.elLabelESP32.textContent = `ESP32 Buzzer: ${statusLabel}`;
        }
        if (this.elStatusESP32) {
          this.elStatusESP32.className = `status-pill ${isConn ? 'online' : 'disconnected'}`;
        }

        // Render recent actuation history
        if (this.esp32ActuationLog) {
          const hist = s.recent_history || [];
          if (hist.length === 0) {
            this.esp32ActuationLog.textContent = 'No hardware trigger attempts yet.';
          } else {
            this.esp32ActuationLog.innerHTML = hist.slice(0, 5).map(h => `
              <div>[${h.timestamp || ''}] <strong>${h.command}</strong> (${h.duration_ms}ms): ${h.success ? '✓ FIRED' : '✕ FAILED'} - ${h.error || h.device_response || ''}</div>
            `).join('');
          }
        }
      }
    } catch (err) {
      console.warn('Could not load ESP32 status:', err);
    }
  }

  async saveESP32Config() {
    const host = this.inputESP32Host ? this.inputESP32Host.value.trim() : '192.168.1.120';
    const port = parseInt(this.inputESP32Port ? this.inputESP32Port.value : '80', 10);
    const cooldown = parseFloat(this.inputESP32Cooldown ? this.inputESP32Cooldown.value : '10');
    const duration = parseInt(this.inputESP32Duration ? this.inputESP32Duration.value : '1500', 10);
    const token = this.inputESP32Token ? this.inputESP32Token.value.trim() : '';

    try {
      const res = await fetch('/api/esp32/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          host: host,
          port: port,
          cooldown_seconds: cooldown,
          default_duration_ms: duration,
          auth_token: token,
        }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        this.showGlobalNotice('ESP32 settings saved.', 'success');
        this.loadESP32Status();
      }
    } catch (err) {
      console.error('Error saving ESP32 config:', err);
    }
  }

  async triggerESP32Pulse() {
    const duration = parseInt(this.inputESP32Duration ? this.inputESP32Duration.value : '1000', 10);
    try {
      const res = await fetch('/api/esp32/buzzer/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ duration_ms: duration }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        const record = data.data;
        if (record.success) {
          this.showGlobalNotice(`Buzzer pulse signal confirmed by ESP32 (${duration}ms)`, 'success');
        } else {
          alert(`Buzzer trigger attempt returned failure:\n${record.error || record.device_response}`);
        }
        this.loadESP32Status();
      }
    } catch (err) {
      console.error('Error triggering buzzer:', err);
    }
  }

  async silenceESP32() {
    try {
      const res = await fetch('/api/esp32/buzzer/silence', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        this.showGlobalNotice('Buzzer silence command transmitted.', 'info');
        this.loadESP32Status();
      }
    } catch (err) {
      console.error('Error silencing buzzer:', err);
    }
  }

  // =========================================================================
  // HISTORICAL INCIDENT LOG & EXPORT
  // =========================================================================

  renderIncidentHistoryTable() {
    if (!this.tbodyIncidentHistory) return;

    const search = (document.getElementById('filter-incident-search')?.value || '').toLowerCase();
    const severity = document.getElementById('filter-incident-severity')?.value || '';
    const status = document.getElementById('filter-incident-status')?.value || '';

    const filtered = this.incidents.filter(inc => {
      if (severity && inc.severity !== severity) return false;
      if (status && inc.state !== status) return false;
      if (search) {
        const matchZone = (inc.primary_zone_name || '').toLowerCase().includes(search);
        const matchId = (inc.incident_id || '').toLowerCase().includes(search);
        const matchVid = (inc.source_video || '').toLowerCase().includes(search);
        if (!matchZone && !matchId && !matchVid) return false;
      }
      return true;
    });

    if (filtered.length === 0) {
      this.tbodyIncidentHistory.innerHTML = `
        <tr>
          <td colspan="10" style="text-align:center; color:var(--text-muted); padding:2rem;">
            No incidents matching current filter criteria.
          </td>
        </tr>
      `;
      return;
    }

    this.tbodyIncidentHistory.innerHTML = filtered.map(inc => {
      const isCrit = inc.severity === 'CRITICAL';
      const isAck = inc.state === 'ACKNOWLEDGED';
      const isRes = inc.state === 'RESOLVED';

      return `
        <tr>
          <td><code>${inc.incident_id}</code></td>
          <td>${inc.source_video || 'CCTV Video'}</td>
          <td>${inc.timestamp_str || ''} (f#${inc.created_at_frame || 0})</td>
          <td><strong>${inc.primary_zone_name}</strong></td>
          <td>${inc.classification} (#${inc.track_id})</td>
          <td><span class="card-badge ${isCrit ? 'badge-planned' : 'badge-warning'}">${inc.severity}</span></td>
          <td><span class="card-badge ${inc.notification_status === 'DELIVERED' ? 'badge-implemented' : 'badge-planned'}">${inc.notification_status || 'PENDING'}</span></td>
          <td><span class="card-badge ${inc.buzzer_status === 'TRIGGERED' ? 'badge-implemented' : 'badge-warning'}">${inc.buzzer_status || 'NOT_CONNECTED'}</span></td>
          <td><span class="status-pill ${isRes ? 'online' : (isAck ? 'unconfigured' : 'disconnected')}">${inc.state}</span></td>
          <td>
            <div style="display:flex; gap:0.25rem;">
              ${!isAck && !isRes ? `
                <button class="btn btn-secondary" style="font-size:0.7rem; padding:0.2rem 0.4rem;" onclick="window.safekidCR.acknowledgeIncident('${inc.incident_id}')">
                  Ack
                </button>
              ` : ''}
              ${!isRes ? `
                <button class="btn btn-secondary" style="font-size:0.7rem; padding:0.2rem 0.4rem; color:var(--emerald-primary);" onclick="window.safekidCR.resolveIncident('${inc.incident_id}')">
                  Resolve
                </button>
              ` : '<span style="font-size:0.75rem; color:var(--text-muted);">Resolved</span>'}
            </div>
          </td>
        </tr>
      `;
    }).join('');
  }

  exportIncidentsCSV() {
    if (this.incidents.length === 0) {
      alert('No incidents to export.');
      return;
    }
    const headers = ['incident_id', 'source_video', 'created_at_ms', 'timestamp_str', 'zone_name', 'classification', 'track_id', 'severity', 'state', 'notification_status', 'buzzer_status'];
    const rows = this.incidents.map(i => [
      i.incident_id,
      `"${i.source_video || ''}"`,
      i.created_at_ms || '',
      `"${i.timestamp_str || ''}"`,
      `"${i.primary_zone_name || ''}"`,
      i.classification || '',
      i.track_id || '',
      i.severity || '',
      i.state || '',
      i.notification_status || '',
      i.buzzer_status || '',
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `safekid_incidents_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  exportIncidentsJSON() {
    if (this.incidents.length === 0) {
      alert('No incidents to export.');
      return;
    }
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(this.incidents, null, 2));
    const link = document.createElement('a');
    link.setAttribute('href', dataStr);
    link.setAttribute('download', `safekid_incidents_${Date.now()}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  // =========================================================================
  // PERIODIC DASHBOARD POLLING
  // =========================================================================

  async refreshDashboard() {
    await Promise.all([
      this.loadJobs(),
      this.loadROIConfig(),
      this.loadIncidents(),
      this.loadNotificationConfig(),
      this.loadESP32Status(),
      this.fetchSystemStats(),
    ]);
  }

  async fetchSystemStats() {
    try {
      const res = await fetch('/api/stats');
      const data = await res.json();
      if (data.status === 'success') {
        const s = data.data;
        if (this.elStatCompletedJobs) this.elStatCompletedJobs.textContent = s.completed_jobs || 0;
      }
    } catch (err) {}
  }

  startPolling() {
    if (this.pollingTimer) clearInterval(this.pollingTimer);
    this.pollingTimer = setInterval(async () => {
      // Check if any job is currently processing or queued
      const hasActive = this.jobs.some(j => ['PROCESSING', 'QUEUED'].includes(j.status));
      if (hasActive) {
        await this.loadJobs();
        await this.loadIncidents();
      }
    }, this.pollIntervalMs);
  }

  showGlobalNotice(msg, type = 'info') {
    console.log(`[SafeKid Notice ${type.toUpperCase()}] ${msg}`);
  }
}

// Instantiate and attach globally
document.addEventListener('DOMContentLoaded', () => {
  window.safekidCR = new SafeKidControlRoom();
  window.safekidCR.init();
});

