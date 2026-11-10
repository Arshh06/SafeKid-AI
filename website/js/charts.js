/**
 * SafeKid AI — Custom Lightweight High-Performance Canvas Charting Engine
 * 100% self-contained, zero external CDN dependencies, fully offline-compatible.
 */

class SafeKidCharts {
  static initCanvas(canvas) {
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    // Set actual canvas size in memory (scaled to account for extra pixel density)
    const width = rect.width || canvas.parentElement.clientWidth || 600;
    const height = rect.height || 320;
    
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    return { ctx, width, height };
  }

  /**
   * Grouped Bar Chart comparing two models across multiple metrics
   */
  static renderGroupedBarChart(canvasId, labels, dataset1, dataset2) {
    const canvas = document.getElementById(canvasId);
    const setup = this.initCanvas(canvas);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 60;
    const padRight = 30;
    const padTop = 40;
    const padBottom = 50;
    const chartW = width - padLeft - padRight;
    const chartH = height - padTop - padBottom;

    ctx.clearRect(0, 0, width, height);

    // Draw background grid lines (0%, 25%, 50%, 75%, 100%)
    ctx.strokeStyle = '#1e2c47';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#64748b';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';

    for (let i = 0; i <= 4; i++) {
      const y = padTop + chartH - (i / 4) * chartH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();
      ctx.fillText(`${i * 25}%`, padLeft - 10, y + 4);
    }

    const numGroups = labels.length;
    const groupW = chartW / numGroups;
    const barW = Math.min(32, (groupW * 0.7) / 2);

    labels.forEach((label, idx) => {
      const groupCenter = padLeft + idx * groupW + groupW / 2;
      const v1 = dataset1.values[idx];
      const v2 = dataset2.values[idx];

      const h1 = (v1 / 100) * chartH;
      const h2 = (v2 / 100) * chartH;

      const x1 = groupCenter - barW - 2;
      const y1 = padTop + chartH - h1;

      const x2 = groupCenter + 2;
      const y2 = padTop + chartH - h2;

      // Draw Bar 1 (YOLOv8n - Emerald)
      ctx.fillStyle = dataset1.color || '#10b981';
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(x1, y1, barW, h1, [4, 4, 0, 0]) : ctx.rect(x1, y1, barW, h1);
      ctx.fill();

      // Draw Bar 2 (D-FINE-N - Cyan)
      ctx.fillStyle = dataset2.color || '#06b6d4';
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(x2, y2, barW, h2, [4, 4, 0, 0]) : ctx.rect(x2, y2, barW, h2);
      ctx.fill();

      // Bar Value Labels
      ctx.fillStyle = '#f1f5f9';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`${v1}%`, x1 + barW / 2, y1 - 6);
      ctx.fillText(`${v2}%`, x2 + barW / 2, y2 - 6);

      // Group X-axis Label
      ctx.fillStyle = '#94a3b8';
      ctx.font = '11px Inter, sans-serif';
      ctx.fillText(label, groupCenter, padTop + chartH + 20);
    });

    // Legend
    this.renderLegend(ctx, width - padRight - 220, 15, [
      { label: dataset1.name, color: dataset1.color || '#10b981' },
      { label: dataset2.name, color: dataset2.color || '#06b6d4' }
    ]);
  }

  /**
   * Horizontal Comparative Progress / Target Bar Chart
   */
  static renderHorizontalBarChart(canvasId, items) {
    const canvas = document.getElementById(canvasId);
    const setup = this.initCanvas(canvas);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 140;
    const padRight = 60;
    const padTop = 30;
    const padBottom = 20;
    const chartW = width - padLeft - padRight;
    const rowHeight = (height - padTop - padBottom) / items.length;

    ctx.clearRect(0, 0, width, height);

    items.forEach((item, idx) => {
      const y = padTop + idx * rowHeight;
      const barH = 22;

      // Label
      ctx.fillStyle = '#f1f5f9';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(item.label, padLeft - 15, y + barH / 2 + 4);

      // Background Track
      ctx.fillStyle = '#1e2c47';
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(padLeft, y, chartW, barH, 4) : ctx.rect(padLeft, y, chartW, barH);
      ctx.fill();

      // Value Bar
      const maxVal = item.max || 100;
      const barW = Math.max(6, (item.value / maxVal) * chartW);
      ctx.fillStyle = item.color || '#06b6d4';
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(padLeft, y, barW, barH, 4) : ctx.rect(padLeft, y, barW, barH);
      ctx.fill();

      // Value Text
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(item.displayValue || `${item.value}%`, padLeft + barW + 10, y + barH / 2 + 4);
    });
  }

  /**
   * Latency Comparison Bar Chart (ms on CPU)
   */
  static renderLatencyChart(canvasId, yoloLatencies, dfineLatencies) {
    const canvas = document.getElementById(canvasId);
    const setup = this.initCanvas(canvas);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 70;
    const padRight = 30;
    const padTop = 40;
    const padBottom = 45;
    const chartW = width - padLeft - padRight;
    const chartH = height - padTop - padBottom;
    const maxMs = 500;

    ctx.clearRect(0, 0, width, height);

    // Grid lines (0, 100, 200, 300, 400, 500 ms)
    ctx.strokeStyle = '#1e2c47';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#64748b';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';

    for (let ms = 0; ms <= maxMs; ms += 100) {
      const y = padTop + chartH - (ms / maxMs) * chartH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();
      ctx.fillText(`${ms} ms`, padLeft - 10, y + 4);
    }

    const metrics = ['Mean', 'Median', 'P90', 'P99'];
    const numGroups = metrics.length;
    const groupW = chartW / numGroups;
    const barW = Math.min(36, (groupW * 0.7) / 2);

    metrics.forEach((metric, idx) => {
      const groupCenter = padLeft + idx * groupW + groupW / 2;
      const v1 = yoloLatencies[idx];
      const v2 = dfineLatencies[idx];

      const h1 = (v1 / maxMs) * chartH;
      const h2 = (v2 / maxMs) * chartH;

      const x1 = groupCenter - barW - 2;
      const y1 = padTop + chartH - h1;

      const x2 = groupCenter + 2;
      const y2 = padTop + chartH - h2;

      // YOLO (Emerald)
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(x1, y1, barW, h1, [4, 4, 0, 0]) : ctx.rect(x1, y1, barW, h1);
      ctx.fill();

      // D-FINE (Amber / Warning color for high latency)
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(x2, y2, barW, h2, [4, 4, 0, 0]) : ctx.rect(x2, y2, barW, h2);
      ctx.fill();

      // Values
      ctx.fillStyle = '#f1f5f9';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`${v1}ms`, x1 + barW / 2, y1 - 6);
      ctx.fillText(`${v2}ms`, x2 + barW / 2, y2 - 6);

      // Label
      ctx.fillStyle = '#94a3b8';
      ctx.font = '11px Inter, sans-serif';
      ctx.fillText(metric, groupCenter, padTop + chartH + 20);
    });

    // Legend
    this.renderLegend(ctx, width - padRight - 260, 15, [
      { label: 'YOLOv8n (Fast Edge Baseline)', color: '#10b981' },
      { label: 'D-FINE-N (Heavy Query Head)', color: '#f59e0b' }
    ]);
  }

  static renderLegend(ctx, x, y, items) {
    let curX = x;
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';

    items.forEach(item => {
      ctx.fillStyle = item.color;
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(curX, y - 8, 12, 12, 2) : ctx.rect(curX, y - 8, 12, 12);
      ctx.fill();

      ctx.fillStyle = '#94a3b8';
      ctx.fillText(item.label, curX + 16, y + 2);
      curX += ctx.measureText(item.label).width + 30;
    });
  }
}

window.SafeKidCharts = SafeKidCharts;

