/* Progressive enhancement: hero wireframe build, copy buttons. The site reads fully without it. */
(() => {
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  document.querySelectorAll("button[data-copy]").forEach((button) => {
    const code = button.parentElement.querySelector("code");
    if (!code || !navigator.clipboard) return;
    button.hidden = false;
    let timer = 0;
    const show = (ok) => {
      button.classList.remove("done", "fail");
      button.classList.add(ok ? "done" : "fail");
      button.textContent = ok ? "Copied" : "Copy failed";
      if (!ok) getSelection().selectAllChildren(code);
      clearTimeout(timer);
      timer = setTimeout(() => {
        button.classList.remove("done", "fail");
        button.textContent = "Copy";
      }, 1800);
    };
    button.addEventListener("click", () => {
      navigator.clipboard.writeText(code.textContent).then(() => show(true), () => show(false));
    });
  });

  const canvas = document.getElementById("titan");
  const hero = document.querySelector(".hero");
  if (!canvas || !hero || !canvas.getContext) return;
  const replay = document.getElementById("replay");
  const setBuilt = (built) => {
    hero.classList.toggle("built", built);
    hero.classList.toggle("drawing", !built);
  };

  fetch(new URL("assets/gm200-wireframe.json", document.baseURI)).then((response) => {
    if (!response.ok) throw new Error(`wireframe ${response.status}`);
    return response.json();
  }).then((d) => {
    const { q, n, e, parts } = d;
    const [fx, fz] = d.fanAxis;
    const isFan = new Uint8Array(n);
    for (const part of d.fan) {
      const start = part === 0 ? 0 : parts[part - 1];
      for (let k = start; k < parts[part]; k++) isFan[k] = 1;
    }
    const ctx = canvas.getContext("2d");
    const yaw0 = 0.62, pitch0 = -0.5, f = 3.2, buildMs = 14000, ox = 0.67, oy = 0.44, buckets = 7;
    const P = new Float32Array(n * 6);
    const tmp = [0, 0, 0];
    const labels = [
      { t: "Blower fan", x: 0.321, y: 0.07, z: -0.039, dx: 90, dy: -70 },
      { t: "GM200 package", x: -0.032, y: -0.04, z: -0.016, dx: -40, dy: 120 },
      { t: "Heatsink fins", x: -0.12, y: 0.06, z: -0.1, dx: 130, dy: -120 },
      { t: "PCIe bracket", x: -0.49, y: 0.06, z: 0.0, dx: 60, dy: -110 },
    ];
    const rot = (x, y, z, yaw, pitch, out, o) => {
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x2 = cy * x + sy * z, z2 = -sy * x + cy * z;
      out[o] = x2; out[o + 1] = cp * y - sp * z2; out[o + 2] = sp * y + cp * z2;
    };
    const sx = (x, z, w, k) => w * ox + x * (k / (f + z));
    const sy = (y, z, h, k) => h * oy - y * (k / (f + z));
    let start = null;
    let built = false;
    let raf = 0;
    const paths = [];

    const frame = (ts) => {
      if (start === null) start = ts;
      const el = ts - start;
      const t = reduce ? 1 : Math.min(1, el / buildMs);
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      const count = Math.floor(ease * n);
      if (t >= 1 && !built) { built = true; setBuilt(true); }
      const yaw = yaw0 + (reduce ? 0 : Math.sin(ts / 16000) * 0.05);
      const spin = reduce ? 0 : ts * 0.0009;
      const cs = Math.cos(spin), sn = Math.sin(spin);
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const k = Math.min(w * 2.0, h * 3.3);
      for (let i = 0; i < count; i++) {
        const b = i * 6;
        let x1 = e[b] / q, y1 = e[b + 1] / q, z1 = e[b + 2] / q, x2 = e[b + 3] / q, y2 = e[b + 4] / q, z2 = e[b + 5] / q;
        if (isFan[i]) {
          const rx1 = x1 - fx, rz1 = z1 - fz, rx2 = x2 - fx, rz2 = z2 - fz;
          x1 = fx + rx1 * cs - rz1 * sn; z1 = fz + rx1 * sn + rz1 * cs;
          x2 = fx + rx2 * cs - rz2 * sn; z2 = fz + rx2 * sn + rz2 * cs;
        }
        rot(x1, y1, z1, yaw, pitch0, P, b);
        rot(x2, y2, z2, yaw, pitch0, P, b + 3);
      }
      let pi = 0;
      while (pi < parts.length && parts[pi] <= count) pi++;
      const partStart = pi === 0 ? 0 : parts[pi - 1];
      const prevStart = pi < 2 ? 0 : parts[pi - 2];
      const settled = count >= n ? n : prevStart;
      for (let bk = 0; bk < buckets; bk++) paths[bk] = new Path2D();
      const seg = (path, j) => {
        path.moveTo(sx(P[j * 6], P[j * 6 + 2], w, k), sy(P[j * 6 + 1], P[j * 6 + 2], h, k));
        path.lineTo(sx(P[j * 6 + 3], P[j * 6 + 5], w, k), sy(P[j * 6 + 4], P[j * 6 + 5], h, k));
      };
      for (let j = 0; j < settled; j++) {
        const z = (P[j * 6 + 2] + P[j * 6 + 5]) * 0.5;
        const depth = Math.max(0, Math.min(1, (z + 1.1) / 2.2));
        seg(paths[Math.min(buckets - 1, Math.floor((1 - depth) * buckets))], j);
      }
      ctx.lineWidth = 1;
      for (let bk = 0; bk < buckets; bk++) {
        ctx.strokeStyle = `rgba(255,255,255,${(0.14 + (bk / (buckets - 1)) * 0.5).toFixed(3)})`;
        ctx.stroke(paths[bk]);
      }
      if (count > settled) {
        const prev = new Path2D(), cur = new Path2D();
        for (let j = settled; j < count; j++) seg(j < partStart ? prev : cur, j);
        ctx.strokeStyle = "rgba(255,212,0,0.45)";
        ctx.stroke(prev);
        ctx.strokeStyle = "#FFD400";
        ctx.shadowColor = "rgba(255,212,0,0.7)";
        ctx.shadowBlur = 5;
        ctx.stroke(cur);
        ctx.shadowBlur = 0;
      }
      const la = reduce ? 1 : Math.max(0, Math.min(1, (el - buildMs + 1200) / 1200));
      if (w >= 860 && la > 0) {
        ctx.font = '600 13px "Archivo", Helvetica, Arial, sans-serif';
        ctx.textBaseline = "middle";
        for (const L of labels) {
          rot(L.x, L.y, L.z, yaw, pitch0, tmp, 0);
          const px = sx(tmp[0], tmp[2], w, k), py = sy(tmp[1], tmp[2], h, k);
          const ex = px + L.dx, ey = py + L.dy;
          ctx.strokeStyle = `rgba(255,255,255,${(0.55 * la).toFixed(3)})`;
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(ex, ey); ctx.lineTo(ex + (L.dx < 0 ? -10 : 10), ey); ctx.stroke();
          ctx.fillStyle = `rgba(255,255,255,${la.toFixed(3)})`;
          ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2); ctx.fill();
          ctx.textAlign = L.dx < 0 ? "right" : "left";
          const tw = ctx.measureText(L.t).width, tx = ex + (L.dx < 0 ? -16 : 16);
          ctx.fillStyle = `rgba(0,0,0,${(0.8 * la).toFixed(3)})`;
          ctx.fillRect(L.dx < 0 ? tx - tw - 6 : tx - 6, ey - 11, tw + 12, 22);
          ctx.fillStyle = `rgba(255,255,255,${la.toFixed(3)})`;
          ctx.fillText(L.t, tx, ey);
        }
      }
      // Stop once the build and labels have settled; pause while the hero is off-screen.
      finished = reduce || el > buildMs + 1200;
      raf = !finished && visible ? requestAnimationFrame(frame) : 0;
    };

    let visible = true;
    let finished = false;
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(([entry]) => {
        visible = entry.isIntersecting;
        if (visible && !raf && !finished) raf = requestAnimationFrame(frame);
      }).observe(canvas);
    }
    window.addEventListener("resize", () => {
      if (finished && !raf) raf = requestAnimationFrame(frame);
    });

    if (!reduce) setBuilt(false);
    raf = requestAnimationFrame(frame);
    if (replay && !reduce) {
      replay.hidden = false;
      replay.addEventListener("click", () => {
        cancelAnimationFrame(raf);
        start = null;
        built = false;
        finished = false;
        setBuilt(false);
        raf = requestAnimationFrame(frame);
      });
    }
  }).catch(() => setBuilt(true));
})();
