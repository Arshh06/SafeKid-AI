/**
 * SafeKid AI — Core Application Controller & State Manager
 */

document.addEventListener('DOMContentLoaded', async () => {
  let evidenceRegistry = null;
  let implementationMatrix = null;

  // 1. Navigation Controller
  const navLinks = document.querySelectorAll('.nav-link, .tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
  const mainNav = document.querySelector('.main-nav');

  function switchTab(targetId) {
    const cleanId = targetId.replace('#', '');
    const activePane = document.getElementById(cleanId);
    if (!activePane) return;

    tabPanes.forEach(pane => pane.classList.remove('active'));
    activePane.classList.add('active');

    navLinks.forEach(link => {
      const href = link.getAttribute('href') || link.getAttribute('data-target');
      if (href === `#${cleanId}` || href === cleanId) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });

    window.location.hash = cleanId;
    window.scrollTo({ top: 0, behavior: 'smooth' });

    if (mainNav && mainNav.classList.contains('mobile-open')) {
      mainNav.classList.remove('mobile-open');
    }

    // Trigger charts on tab render
    renderTabCharts(cleanId);
  }

  // Handle click on nav items
  document.addEventListener('click', (e) => {
    const target = e.target.closest('[data-target], a[href^="#"]');
    if (target) {
      const href = target.getAttribute('href') || target.getAttribute('data-target');
      if (href && href.startsWith('#')) {
        e.preventDefault();
        switchTab(href);
      }
    }
  });

  // Mobile menu toggle
  if (mobileMenuBtn && mainNav) {
    mobileMenuBtn.addEventListener('click', () => {
      mainNav.classList.toggle('mobile-open');
    });
  }

  // 2. Load Evidence Registry Data
  try {
    const regRes = await fetch('data/evidence_registry.json');
    if (regRes.ok) {
      evidenceRegistry = await regRes.json();
      console.log('SafeKid AI: Evidence registry loaded successfully.');
    }
  } catch (err) {
    console.warn('Could not fetch external registry JSON; using embedded state.', err);
  }

  // 3. Load Implementation Matrix Data
  try {
    const matRes = await fetch('data/implementation_matrix.json');
    if (matRes.ok) {
      implementationMatrix = await matRes.json();
      renderImplementationTable(implementationMatrix);
    }
  } catch (err) {
    console.warn('Could not fetch implementation matrix JSON.', err);
  }

  // 4. Render Dynamic Elements
  renderVisualGallery();
  renderEvidenceInspector(evidenceRegistry);

  // Initialize view from URL hash or default to home
  const initialHash = window.location.hash || '#home';
  switchTab(initialHash);

  // Chart Rendering Helper
  function renderTabCharts(tabId) {
    // Give browser a frame to layout canvas dimensions
    setTimeout(() => {
      if (tabId === 'model-comparison') {
        // Grouped Bar: Accuracy Metrics
        SafeKidCharts.renderGroupedBarChart(
          'chart-accuracy-comparison',
          ['Precision', 'Recall', 'F1-Score', 'AP@0.50', 'AP@0.75', 'mAP@50:95'],
          { name: 'YOLOv8n (Baseline)', values: [87.80, 46.64, 60.92, 54.47, 17.81, 24.97], color: '#10b981' },
          { name: 'D-FINE-N (Research)', values: [41.71, 56.60, 48.03, 53.02, 15.72, 23.18], color: '#06b6d4' }
        );

        // Horizontal: Subgroup Comparison
        SafeKidCharts.renderHorizontalBarChart('chart-subgroup-bars', [
          { label: 'YOLOv8n: Child Recall (949/1936)', value: 49.02, max: 100, color: '#10b981', displayValue: '49.02%' },
          { label: 'D-FINE-N: Child Recall (1053/1936)', value: 54.39, max: 100, color: '#06b6d4', displayValue: '54.39%' },
          { label: 'YOLOv8n: Small (<32px) Recall (15/459)', value: 3.27, max: 100, color: '#3b82f6', displayValue: '3.27%' },
          { label: 'D-FINE-N: Small (<32px) Recall (106/459)', value: 23.09, max: 100, color: '#38bdf8', displayValue: '23.09%' }
        ]);
      } else if (tabId === 'small-pedestrian') {
        SafeKidCharts.renderHorizontalBarChart('chart-small-pedestrian', [
          { label: 'YOLOv8n (<32px Recall)', value: 3.27, max: 30, color: '#10b981', displayValue: '3.27% (15/459)' },
          { label: 'D-FINE-N (<32px Recall)', value: 23.09, max: 30, color: '#06b6d4', displayValue: '23.09% (106/459)' }
        ]);
      } else if (tabId === 'child-evaluation') {
        SafeKidCharts.renderHorizontalBarChart('chart-child-evaluation', [
          { label: 'YOLOv8n Child Recall (All-GT)', value: 49.02, max: 100, color: '#10b981', displayValue: '49.02% (949/1936)' },
          { label: 'D-FINE-N Child Recall (All-GT)', value: 54.39, max: 100, color: '#06b6d4', displayValue: '54.39% (1053/1936)' },
          { label: 'YOLOv8n Adult Recall (Reference)', value: 67.52, max: 100, color: '#64748b', displayValue: '67.52% (3197/4735)' },
          { label: 'D-FINE-N Adult Recall (Reference)', value: 72.78, max: 100, color: '#94a3b8', displayValue: '72.78% (3446/4735)' }
        ]);
      } else if (tabId === 'hardware') {
        SafeKidCharts.renderLatencyChart(
          'chart-latency-percentiles',
          [106.43, 101.78, 122.64, 181.41],
          [289.42, 280.81, 332.63, 471.90]
        );
      }
    }, 60);
  }

  // 5. Render Implementation Table
  function renderImplementationTable(items) {
    const tbody = document.getElementById('implementation-tbody');
    if (!tbody || !items) return;

    tbody.innerHTML = items.map(item => {
      let badgeClass = 'badge-implemented';
      if (item.Status.includes('PLANNED')) badgeClass = 'badge-planned';
      else if (item.Status.includes('EVALUATED')) badgeClass = 'badge-evaluated';
      else if (item.Status.includes('TRAINED')) badgeClass = 'badge-baseline';

      return `
        <tr>
          <td>
            <strong>${item.Module_Component}</strong>
            <div style="font-size:0.75rem; color:var(--text-muted); font-family:var(--font-mono);">${item.Source_Path}</div>
          </td>
          <td><span class="card-badge ${badgeClass}">${item.Status}</span></td>
          <td style="font-size:0.8rem;">${item.Test_Coverage}</td>
          <td style="font-size:0.8rem; color:var(--text-secondary);">${item.Experimental_Evaluation}</td>
        </tr>
      `;
    }).join('');
  }

  // 6. Visual Gallery Setup
  function renderVisualGallery() {
    const galleryContainer = document.getElementById('gallery-items');
    if (!galleryContainer) return;

    const plates = [
      {
        filename: 'comparison_video_0005_f0102_correct_detections_children_adults.jpg',
        title: 'JAAD video_0005 (Frame 102)',
        meta: 'Resolution: 1920x1080 | Daylight Crossing',
        desc: 'Both detectors successfully capture walking child and adult pedestrians. D-FINE-N produces tighter box on smaller child silhouette but slightly looser margins on adult.'
      },
      {
        filename: 'comparison_video_0105_f0143_crossing_child_mid_street.jpg',
        title: 'JAAD video_0105 (Frame 143)',
        meta: 'Resolution: 1920x1080 | Mid-Street Crossing',
        desc: 'Child pedestrian isolated in roadway center. Both YOLOv8n and D-FINE-N achieve high IoU (>0.75) against ground truth box.'
      },
      {
        filename: 'comparison_video_0128_f0322_partially_occluded_child_curb.jpg',
        title: 'JAAD video_0128 (Frame 322)',
        meta: 'Resolution: 1920x1080 | Partial Occlusion',
        desc: 'Child partially occluded behind roadside barrier. D-FINE-N queries recover upper body silhouette where YOLOv8n confidence drops below threshold.'
      },
      {
        filename: 'comparison_video_0152_f0063_dense_crowd_inter_pedestrian_occlusion.jpg',
        title: 'JAAD video_0152 (Frame 63)',
        meta: 'Resolution: 1920x1080 | Dense Crowd Occlusion',
        desc: 'Severe inter-pedestrian occlusion in pedestrian crossing. Demonstrates boundary competition between adjacent child and adult instances.'
      },
      {
        filename: 'comparison_video_0305_f0080_background_false_positives_comparison.jpg',
        title: 'JAAD video_0305 (Frame 80)',
        meta: 'Resolution: 1920x1080 | Background Suppression',
        desc: 'Demonstrates false-positive contrast. D-FINE-N query decoders trigger on complex foliage background while YOLOv8n anchor-free head correctly suppresses background.'
      },
      {
        filename: 'comparison_video_0317_f0166_test_split_two_children_crossing.jpg',
        title: 'JAAD video_0317 (Frame 166)',
        meta: 'Resolution: 1920x1080 | Two Children Crossing',
        desc: 'Two children crossing together at distant curb. Both detectors detect lead child; D-FINE-N captures second trailing child at small scale.'
      },
      {
        filename: 'comparison_virat_sample_f0030.jpg',
        title: 'VIRAT S_000001 (Frame 30)',
        meta: 'Resolution: 1920x1080 | Elevated CCTV (30° Angle)',
        desc: 'Elevated CCTV perspective sanity test. Bounding box lower-body fragmentation confirmed; lamp-post hallucination claim formally retracted.'
      }
    ];

    galleryContainer.innerHTML = plates.map((p, idx) => `
      <div class="gallery-card" onclick="openPlateModal('${p.filename}', '${p.title}', '${p.meta}', '${p.desc.replace(/'/g, "\\'")}')">
        <div class="gallery-img-wrapper">
          <img src="assets/plates/${p.filename}" alt="${p.title}" loading="lazy">
        </div>
        <div class="gallery-content">
          <div class="gallery-title">${p.title}</div>
          <div class="gallery-meta">${p.meta}</div>
          <div class="gallery-desc">${p.desc}</div>
        </div>
      </div>
    `).join('');
  }

  // 7. Interactive Evidence Registry Inspector
  function renderEvidenceInspector(registry) {
    const listContainer = document.getElementById('registry-metric-list');
    if (!listContainer || !registry || !registry.authoritative_metrics) return;

    listContainer.innerHTML = registry.authoritative_metrics.map((m, idx) => `
      <div class="card" style="margin-bottom: 1rem; cursor: pointer;" onclick="toggleRegistryDetail(${idx})">
        <div class="card-header" style="margin-bottom: 0.5rem;">
          <div style="font-weight: 600; color: #fff; font-size: 1rem;">${m.metric_name}</div>
          <span class="card-badge ${m.model_name.includes('YOLO') ? 'badge-baseline' : 'badge-candidate'}">${m.model_name}</span>
        </div>
        <div style="display: flex; align-items: baseline; gap: 1rem; margin-bottom: 0.5rem;">
          <span style="font-size: 1.5rem; font-weight: 700; color: var(--cyan-primary);">${m.display_value}</span>
          <span style="font-size: 0.8rem; color: var(--text-secondary);">${m.evaluation_dataset} (${m.evaluated_frames} frames)</span>
        </div>
        <div style="font-size: 0.85rem; color: var(--text-secondary);">${m.exact_permitted_website_wording}</div>
        <div id="reg-detail-${idx}" style="display: none; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-color); font-size: 0.8rem; color: var(--text-muted);">
          <div><strong>Calculation:</strong> <code>${m.calculation_method}</code></div>
          <div><strong>Source Prediction:</strong> <code>${m.source_prediction_artifact}</code></div>
          <div><strong>Source Script:</strong> <code>${m.source_script}</code></div>
          <div style="margin-top: 0.35rem; color: var(--amber-primary);"><strong>Limitation:</strong> ${m.qualification_limitation}</div>
        </div>
      </div>
    `).join('');
  }
});

// Modal inspection helper
window.openPlateModal = function(filename, title, meta, desc) {
  const modal = document.getElementById('plate-modal');
  if (!modal) return;
  document.getElementById('modal-img').src = `assets/plates/${filename}`;
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-meta').textContent = meta;
  document.getElementById('modal-desc').textContent = desc;
  modal.style.display = 'flex';
};

window.closePlateModal = function() {
  const modal = document.getElementById('plate-modal');
  if (modal) modal.style.display = 'none';
};

window.toggleRegistryDetail = function(idx) {
  const el = document.getElementById(`reg-detail-${idx}`);
  if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
};

