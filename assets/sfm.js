/* sfm.js — 在瀏覽器裡跑的小型增量式 SfM (Structure from Motion)
 *
 * 管線: AKAZE 特徵 (OpenCV.js) -> knn 匹配 + ratio test
 *       -> 八點法本質矩陣 + RANSAC (純 JS, Jacobi 特徵分解)
 *       -> 本質矩陣分解 + cheirality 選解 -> DLT 三角化
 *       -> solvePnPRansac 逐張註冊 (OpenCV.js) -> 三角化長點 -> 重投影濾除
 * 註: 官方文件版 opencv.js 的白名單拿掉了 findEssentialMat / recoverPose /
 *     triangulatePoints / SVD, 所以幾何部分全部用純 JS 補上 (已用合成場景驗證).
 * 輸出座標: 第 1 台相機為原點的 OpenCV 慣例 (Y 向下); 轉 splat 時翻 Y/Z.
 */
(function (global) {
  "use strict";

  const WORK_MAX_DIM = 1280;
  const RATIO = 0.85; // binary descriptors (AKAZE) need a looser ratio than SIFT's 0.75
  const MIN_PAIR_MATCHES = 24;
  const MIN_INIT_INLIERS = 55;
  const MIN_PNP_INLIERS = 24;
  const MAX_REPROJ_PX = 4.5;

  const sleep = () => new Promise((r) => setTimeout(r, 0));
  const focalFor = (w, h) => 0.9 * Math.max(w, h);

  /* ---------------- 小型線性代數 (純 JS) ---------------- */
  function jacobiEigen(Ain) {
    // 對稱矩陣特徵分解. 回傳 { values: 降冪, vectors: 行主序, 第 k 個特徵向量 = column k }
    const n = Ain.length;
    const A = Ain.map((r) => Float64Array.from(r));
    const V = [];
    for (let i = 0; i < n; i++) { V.push(new Float64Array(n)); V[i][i] = 1; }
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < 1e-20) break;
      for (let p = 0; p < n - 1; p++) {
        for (let q = p + 1; q < n; q++) {
          const apq = A[p][q];
          if (Math.abs(apq) < 1e-14) continue;
          const tau = (A[q][q] - A[p][p]) / (2 * apq);
          const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
          const c = 1 / Math.sqrt(1 + t * t), s = t * c;
          for (let k = 0; k < n; k++) {
            const akp = A[k][p], akq = A[k][q];
            A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
          }
          for (let k = 0; k < n; k++) {
            const apk = A[p][k], aqk = A[q][k];
            A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
          }
          for (let k = 0; k < n; k++) {
            const vkp = V[k][p], vkq = V[k][q];
            V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
          }
        }
      }
    }
    const order = [...Array(n).keys()].sort((a, b) => A[b][b] - A[a][a]);
    return {
      values: order.map((i) => A[i][i]),
      vectors: order.map((col) => { const v = []; for (let r = 0; r < n; r++) v.push(V[r][col]); return v; }),
    };
  }
  const mm3 = (a, b) => { const o = new Array(9); for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c]; return o; };
  const tr3 = (a) => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
  const mv3 = (a, v) => [a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2], a[6] * v[0] + a[7] * v[1] + a[8] * v[2]];
  const det3 = (a) => a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const colsToMat = (c) => [c[0][0], c[1][0], c[2][0], c[0][1], c[1][1], c[2][1], c[0][2], c[1][2], c[2][2]]; // vectors are columns

  function symEigFromATA(rows, n) {
    const AtA = Array.from({ length: n }, () => new Float64Array(n));
    for (const row of rows) for (let r = 0; r < n; r++) { const rv = row[r]; if (rv === 0) continue; for (let c = 0; c < n; c++) AtA[r][c] += rv * row[c]; }
    return jacobiEigen(AtA);
  }

  /* ---------------- 幾何核心 ---------------- */
  function triangulateDLT(Pa, Pb, a, b) {
    const rows = [
      [Pa[0] - a[0] * Pa[8], Pa[1] - a[0] * Pa[9], Pa[2] - a[0] * Pa[10], Pa[3] - a[0] * Pa[11]],
      [Pa[4] - a[1] * Pa[8], Pa[5] - a[1] * Pa[9], Pa[6] - a[1] * Pa[10], Pa[7] - a[1] * Pa[11]],
      [Pb[0] - b[0] * Pb[8], Pb[1] - b[0] * Pb[9], Pb[2] - b[0] * Pb[10], Pb[3] - b[0] * Pb[11]],
      [Pb[4] - b[1] * Pb[8], Pb[5] - b[1] * Pb[9], Pb[6] - b[1] * Pb[10], Pb[7] - b[1] * Pb[11]],
    ];
    const eig = symEigFromATA(rows, 4);
    const v = eig.vectors[3];
    if (Math.abs(v[3]) < 1e-12) return null;
    return [v[0] / v[3], v[1] / v[3], v[2] / v[3]];
  }
  function projMatArr(f, cx, cy, R, t) {
    const Rt = [R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2]];
    const K = [f, 0, cx, 0, f, cy, 0, 0, 1];
    const o = new Array(12);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) o[r * 4 + c] = K[r * 3] * Rt[c] + K[r * 3 + 1] * Rt[4 + c] + K[r * 3 + 2] * Rt[8 + c];
    return o;
  }
  const depthOf = (R, t, X) => R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  function projectPx(f, cx, cy, R, t, X) {
    const x = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
    const y = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
    const z = depthOf(R, t, X);
    if (z <= 1e-9) return null;
    return [f * x / z + cx, f * y / z + cy];
  }

  function eightPointE(pairs) {
    // pairs: [[[bx1,by1],[bx2,by2]], ...] in Hartley-normalized bearing coords
    const rows = pairs.map(([[x1, y1], [x2, y2]]) => [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]);
    const eig = symEigFromATA(rows, 9);
    return eig.vectors[8];
  }
  function hartley(pts) {
    let mx = 0, my = 0;
    for (const p of pts) { mx += p[0]; my += p[1]; }
    mx /= pts.length; my /= pts.length;
    let d = 0;
    for (const p of pts) d += Math.hypot(p[0] - mx, p[1] - my);
    d /= pts.length;
    const s = Math.SQRT2 / (d || 1);
    return { T: [s, 0, -s * mx, 0, s, -s * my, 0, 0, 1], apply: (p) => [(p[0] - mx) * s, (p[1] - my) * s] };
  }
  function enforceRank2(E) {
    const EEt = mm3(E, tr3(E)), EtE = mm3(tr3(E), E);
    const eigU = jacobiEigen([[EEt[0], EEt[1], EEt[2]], [EEt[3], EEt[4], EEt[5]], [EEt[6], EEt[7], EEt[8]]]);
    const eigV = jacobiEigen([[EtE[0], EtE[1], EtE[2]], [EtE[3], EtE[4], EtE[5]], [EtE[6], EtE[7], EtE[8]]]);
    const u3 = eigU.vectors[2], v3 = eigV.vectors[2];
    const s3 = dot(mv3(E, v3), u3);
    const out = E.slice();
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[r * 3 + c] -= s3 * u3[r] * v3[c];
    return out;
  }
  function sampson(E, x1, x2) {
    const Ex1 = mv3(E, [x1[0], x1[1], 1]);
    const Etx2 = mv3(tr3(E), [x2[0], x2[1], 1]);
    const num = dot([x2[0], x2[1], 1], Ex1);
    const den = Ex1[0] * Ex1[0] + Ex1[1] * Ex1[1] + Etx2[0] * Etx2[0] + Etx2[1] * Etx2[1];
    return den < 1e-18 ? 1e9 : (num * num) / den;
  }
  function estimateEssential(pts1, pts2, f1, c1, f2, c2, itersOpt) {
    const b1 = pts1.map((p) => [(p[0] - c1[0]) / f1, (p[1] - c1[1]) / f1]);
    const b2 = pts2.map((p) => [(p[0] - c2[0]) / f2, (p[1] - c2[1]) / f2]);
    const n = b1.length;
    const thr = Math.pow(1.6 / Math.min(f1, f2), 2);
    function refit(inlIdx) {
      if (inlIdx.length < 8) return null;
      const h1 = hartley(inlIdx.map((i) => b1[i])), h2 = hartley(inlIdx.map((i) => b2[i]));
      let E = eightPointE(inlIdx.map((i) => [h1.apply(b1[i]), h2.apply(b2[i])]));
      E = enforceRank2(mm3(mm3(tr3(h2.T), E), h1.T));
      const inl = [];
      for (let i = 0; i < n; i++) if (sampson(E, b1[i], b2[i]) < thr) inl.push(i);
      return { E, inl };
    }
    let bestInl = null, bestE = null;
    const ITERS = itersOpt || 500;
    for (let it = 0; it < ITERS; it++) {
      const pick = new Set();
      while (pick.size < 8) pick.add((Math.random() * n) | 0);
      const idx = [...pick];
      const h1 = hartley(idx.map((i) => b1[i])), h2 = hartley(idx.map((i) => b2[i]));
      const pairs = idx.map((i) => [h1.apply(b1[i]), h2.apply(b2[i])]);
      let E = eightPointE(pairs);
      E = mm3(mm3(tr3(h2.T), E), h1.T);
      E = enforceRank2(E);
      const inl = [];
      for (let i = 0; i < n; i++) if (sampson(E, b1[i], b2[i]) < thr) inl.push(i);
      if (!bestInl || inl.length > bestInl.length) {
        bestInl = inl; bestE = E;
        // local optimization: refit on the inlier set and re-score (LO-RANSAC)
        const lo = refit(inl);
        if (lo && lo.inl.length > bestInl.length) { bestInl = lo.inl; bestE = lo.E; }
        if (bestInl.length > 0.85 * n) break;
      }
    }
    if (!bestInl || bestInl.length < 8) return null;
    const fin = refit(bestInl);
    if (fin && fin.inl.length >= bestInl.length) return { E: fin.E, inliers: fin.inl, b1, b2 };
    return { E: bestE, inliers: bestInl, b1, b2 };
  }
  function decomposeEssential(E, b1, b2, inlIdx, f1, c1, f2, c2) {
    const EEt = mm3(E, tr3(E)), EtE = mm3(tr3(E), E);
    const eigU = jacobiEigen([[EEt[0], EEt[1], EEt[2]], [EEt[3], EEt[4], EEt[5]], [EEt[6], EEt[7], EEt[8]]]);
    const eigV = jacobiEigen([[EtE[0], EtE[1], EtE[2]], [EtE[3], EtE[4], EtE[5]], [EtE[6], EtE[7], EtE[8]]]);
    let [u1, u2, u3] = eigU.vectors;
    const [v1, v2, v3raw] = eigV.vectors;
    let v3 = v3raw;
    if (dot(mv3(E, v1), u1) < 0) { u1 = u1.map((x) => -x); }
    if (dot(mv3(E, v2), u2) < 0) { u2 = u2.map((x) => -x); }
    let U = colsToMat([u1, u2, u3]);
    if (det3(U) < 0) { u3 = u3.map((x) => -x); U = colsToMat([u1, u2, u3]); }
    let V = colsToMat([v1, v2, v3]);
    if (det3(V) < 0) { v3 = v3.map((x) => -x); V = colsToMat([v1, v2, v3]); }
    const W = [0, -1, 0, 1, 0, 0, 0, 0, 1];
    let Ra = mm3(mm3(U, W), tr3(V));
    let Rb = mm3(mm3(U, tr3(W)), tr3(V));
    if (det3(Ra) < 0) Ra = Ra.map((x) => -x);
    if (det3(Rb) < 0) Rb = Rb.map((x) => -x);
    const t = u3;
    const P1 = projMatArr(f1, c1[0], c1[1], [1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]);
    let best = null;
    for (const R of [Ra, Rb]) {
      for (const sgn of [1, -1]) {
        const tt = t.map((x) => x * sgn);
        const P2 = projMatArr(f2, c2[0], c2[1], R, tt);
        let pos = 0;
        for (const i of inlIdx) {
          const X = triangulateDLT(P1, P2, [(b1[i][0] * f1 + c1[0]), (b1[i][1] * f1 + c1[1])], [(b2[i][0] * f2 + c2[0]), (b2[i][1] * f2 + c2[1])]);
          if (X && X[2] > 0 && depthOf(R, tt, X) > 0) pos++;
        }
        if (!best || pos > best.pos) best = { R, t: tt, pos };
      }
    }
    return best;
  }

  /* ---------------- 特徵擷取 / 匹配 (OpenCV.js) ---------------- */
  async function extractFeatures(images, onProgress) {
    const akaze = new cv.AKAZE();
    if (akaze.setThreshold) akaze.setThreshold(0.0006);
    const feats = [];
    for (let i = 0; i < images.length; i++) {
      const img = images[i];
      let imageData, w, h;
      if (img.imageData) {
        imageData = img.imageData; w = imageData.width; h = imageData.height;
      } else {
        const scale = Math.min(1, WORK_MAX_DIM / Math.max(img.width, img.height));
        w = Math.round(img.width * scale); h = Math.round(img.height * scale);
        const canvas = document.createElement("canvas");
        canvas.width = w; canvas.height = h;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(img.bitmap, 0, 0, w, h);
        imageData = ctx.getImageData(0, 0, w, h);
      }
      const rgba = cv.matFromImageData(imageData);
      const gray = new cv.Mat();
      cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
      const kps = new cv.KeyPointVector();
      const desc = new cv.Mat();
      const noMask = new cv.Mat();
      akaze.detectAndCompute(gray, noMask, kps, desc);
      noMask.delete();
      let pts = [];
      const responses = [];
      for (let k = 0; k < kps.size(); k++) { const kp = kps.get(k); pts.push([kp.pt.x, kp.pt.y]); responses.push(kp.response); }
      let descF = desc;
      const MAX_KP = 9000;
      if (pts.length > MAX_KP && desc.rows === pts.length) {
        const order = pts.map((_, k) => k).sort((a, b) => responses[b] - responses[a]).slice(0, MAX_KP).sort((a, b) => a - b);
        const nd = new cv.Mat(order.length, desc.cols, desc.type());
        const srcData = desc.data, dstData = nd.data, colsB = desc.cols;
        order.forEach((srcRow, k) => { dstData.set(srcData.subarray(srcRow * colsB, srcRow * colsB + colsB), k * colsB); });
        pts = order.map((k) => pts[k]);
        descF = nd; desc.delete();
      }
      const natW = img.imageData ? img.imageData.width : img.width;
      const fWork = img.focalPx ? img.focalPx * (w / natW) : focalFor(w, h);
      feats.push({ name: img.name, w, h, f: fWork, fBase: fWork, pts, desc: descF, imageData });
      rgba.delete(); gray.delete(); kps.delete();
      onProgress && onProgress("features", i + 1, images.length);
      await sleep();
    }
    akaze.delete();
    return feats;
  }
  function matchPair(fi, fj) {
    const bf = new cv.BFMatcher(cv.NORM_HAMMING, false);
    const knn = new cv.DMatchVectorVector();
    bf.knnMatch(fi.desc, fj.desc, knn, 2);
    const good = [];
    for (let k = 0; k < knn.size(); k++) {
      const pair = knn.get(k);
      if (pair.size() < 2) continue;
      const m = pair.get(0), nn = pair.get(1);
      if (m.distance < RATIO * nn.distance) good.push({ q: m.queryIdx, t: m.trainIdx, d: m.distance });
    }
    knn.delete(); bf.delete();
    return good;
  }

  /* ---------------- 主流程 ---------------- */
  async function runSfM(images, onProgress) {
    if (typeof cv === "undefined" || !cv.Mat) throw new Error("OpenCV 尚未載入完成");
    if (images.length < 5) throw new Error("照片太少: 至少需要 5 張 (建議 12-20 張)");

    const feats = await extractFeatures(images, onProgress);
    const n = feats.length;

    const pairMatches = new Map();
    const pairKey = (a, b) => (a < b ? a + "_" + b : b + "_" + a);
    function verifyPair(fi, fj, matches) {
      // two-view geometric verification: keep only essential-matrix inliers
      const sub = matches.length > 700 ? matches.filter((_, k) => k % Math.ceil(matches.length / 700) === 0) : matches;
      const est = estimateEssential(
        sub.map((m) => fi.pts[m.q]), sub.map((m) => fj.pts[m.t]),
        fi.f, [fi.w / 2, fi.h / 2], fj.f, [fj.w / 2, fj.h / 2], 200);
      if (!est) return matches;
      const thr = Math.pow(1.8 / Math.min(fi.f, fj.f), 2);
      const kept = [];
      for (const m of matches) {
        const b1 = [(fi.pts[m.q][0] - fi.w / 2) / fi.f, (fi.pts[m.q][1] - fi.h / 2) / fi.f];
        const b2 = [(fj.pts[m.t][0] - fj.w / 2) / fj.f, (fj.pts[m.t][1] - fj.h / 2) / fj.f];
        if (sampson(est.E, b1, b2) < thr) kept.push(m);
      }
      // 窄基線影像對的本質矩陣先天不穩, 驗證若砍掉太多就視為「無法判定」,
      // 保留原始匹配, 交給下游的 PnP RANSAC + 重投影/視差濾除把關.
      if (kept.length >= Math.max(MIN_PAIR_MATCHES, matches.length * 0.35)) return kept;
      return matches;
    }
    const pairList = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const d = j - i;
      if (n <= 10 || d <= 4 || d % 5 === 0) pairList.push([i, j]);
    }
    let pairDone = 0;
    for (const [i, j] of pairList) {
      const g = matchPair(feats[i], feats[j]);
      if (g.length >= MIN_PAIR_MATCHES) {
        const v = verifyPair(feats[i], feats[j], g);
        if (v) pairMatches.set(pairKey(i, j), { i, j, matches: v });
      }
      pairDone++;
      if (pairDone % 4 === 0 || pairDone === pairList.length) onProgress && onProgress("matching", pairDone, pairList.length);
      await sleep();
    }
    if (pairMatches.size === 0) throw new Error("影像之間找不到足夠的共同特徵 — 請確認照片重疊率夠高、畫面有紋理");
    function oriented(a, b) {
      const rec = pairMatches.get(pairKey(a, b));
      if (!rec) return null;
      if (rec.i === a) return rec.matches;
      return rec.matches.map((m) => ({ q: m.t, t: m.q, d: m.d }));
    }

    const poses = new Array(n).fill(null);
    const kpPoint = feats.map((f) => new Int32Array(f.pts.length).fill(-1));
    const points = [];
    function sampleColor(p, img, kp) {
      const f = feats[img], d = f.imageData.data;
      const px = Math.min(f.w - 1, Math.max(0, Math.round(f.pts[kp][0])));
      const py = Math.min(f.h - 1, Math.max(0, Math.round(f.pts[kp][1])));
      const off = (py * f.w + px) * 4;
      p.rgbSum[0] += d[off]; p.rgbSum[1] += d[off + 1]; p.rgbSum[2] += d[off + 2]; p.rgbCnt++;
    }
    function addPoint(X, obsList) {
      const idx = points.length;
      const p = { xyz: X, obs: obsList, rgbSum: [0, 0, 0], rgbCnt: 0 };
      for (const o of obsList) { kpPoint[o.img][o.kp] = idx; sampleColor(p, o.img, o.kp); }
      points.push(p);
      return idx;
    }
    function attach(img, kp, pIdx) {
      kpPoint[img][kp] = pIdx;
      points[pIdx].obs.push({ img, kp });
      sampleColor(points[pIdx], img, kp);
    }
    function triangulatePair(a, b) {
      const matches = oriented(a, b);
      if (!matches) return 0;
      const pa = poses[a], pb = poses[b];
      const Ca = camCenter(pa), Cb = camCenter(pb);
      const fa = feats[a].f, fb = feats[b].f;
      const Pa = projMatArr(fa, feats[a].w / 2, feats[a].h / 2, pa.R, pa.t);
      const Pb = projMatArr(fb, feats[b].w / 2, feats[b].h / 2, pb.R, pb.t);
      let created = 0;
      for (const m of matches) {
        const ia = kpPoint[a][m.q], ib = kpPoint[b][m.t];
        if (ia >= 0 && ib >= 0) continue;
        if (ia >= 0) { attach(b, m.t, ia); continue; }
        if (ib >= 0) { attach(a, m.q, ib); continue; }
        const ptA = feats[a].pts[m.q], ptB = feats[b].pts[m.t];
        const X = triangulateDLT(Pa, Pb, ptA, ptB);
        if (!X) continue;
        if (depthOf(pa.R, pa.t, X) <= 0 || depthOf(pb.R, pb.t, X) <= 0) continue;
        { // parallax: rays from the two camera centers must differ by >= ~1.7 deg
          const d1 = [X[0] - Ca[0], X[1] - Ca[1], X[2] - Ca[2]];
          const d2 = [X[0] - Cb[0], X[1] - Cb[1], X[2] - Cb[2]];
          const n1 = Math.hypot(d1[0], d1[1], d1[2]), n2 = Math.hypot(d2[0], d2[1], d2[2]);
          if (n1 < 1e-9 || n2 < 1e-9) continue;
          const cosang = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (n1 * n2);
          if (cosang > 0.99955) continue;
        }
        const pra = projectPx(fa, feats[a].w / 2, feats[a].h / 2, pa.R, pa.t, X);
        const prb = projectPx(fb, feats[b].w / 2, feats[b].h / 2, pb.R, pb.t, X);
        if (!pra || !prb) continue;
        if (Math.hypot(pra[0] - ptA[0], pra[1] - ptA[1]) > MAX_REPROJ_PX) continue;
        if (Math.hypot(prb[0] - ptB[0], prb[1] - ptB[1]) > MAX_REPROJ_PX) continue;
        addPoint(X, [{ img: a, kp: m.q }, { img: b, kp: m.t }]);
        created++;
      }
      return created;
    }

    function camCenter(p) {
      return [
        -(p.R[0] * p.t[0] + p.R[3] * p.t[1] + p.R[6] * p.t[2]),
        -(p.R[1] * p.t[0] + p.R[4] * p.t[1] + p.R[7] * p.t[2]),
        -(p.R[2] * p.t[0] + p.R[5] * p.t[1] + p.R[8] * p.t[2]),
      ];
    }


    /* ---- Bundle Adjustment (LM, Schur complement, poses+points, fixed intrinsics) ---- */
    function expSO3(w) {
      const th = Math.hypot(w[0], w[1], w[2]);
      const Kx = [0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0];
      if (th < 1e-10) return [1 + Kx[0], Kx[1], Kx[2], Kx[3], 1 + Kx[4], Kx[5], Kx[6], Kx[7], 1 + Kx[8]];
      const a = Math.sin(th) / th, b = (1 - Math.cos(th)) / (th * th);
      const K2 = mm3(Kx, Kx);
      const out = new Array(9);
      for (let i = 0; i < 9; i++) out[i] = (i % 4 === 0 ? 1 : 0) + a * Kx[i] + b * K2[i];
      return out;
    }
    function solveDense(A, b) {
      const n = b.length;
      const M = A.map((row, i) => [...row, b[i]]);
      for (let col = 0; col < n; col++) {
        let piv = col;
        for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
        if (Math.abs(M[piv][col]) < 1e-14) continue;
        [M[col], M[piv]] = [M[piv], M[col]];
        const d = M[col][col];
        for (let r = 0; r < n; r++) {
          if (r === col) continue;
          const f = M[r][col] / d;
          if (f === 0) continue;
          for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
        }
      }
      return M.map((row, i) => row[n] / (Math.abs(row[i]) < 1e-14 ? 1e-14 : row[i]));
    }
    function bundleAdjust(iters) {
      const camIdx = [];
      for (let i = 0; i < n; i++) if (poses[i]) camIdx.push(i);
      if (camIdx.length < 2 || points.length < 10) return;
      const fixedCam = camIdx[0];
      const freeCams = camIdx.filter((c) => c !== fixedCam);
      const camSlot = new Map(freeCams.map((c, k) => [c, k]));
      const nc = freeCams.length * 6;
      if (nc === 0) return;
      // working copies: R, C per camera; X per point
      const Rc = new Map(), Cc = new Map();
      for (const c of camIdx) { Rc.set(c, poses[c].R.slice()); Cc.set(c, camCenter(poses[c])); }
      const Xw = points.map((p) => p.xyz.slice());
      let lambda = 1e-3;
      const obsOf = points.map((p) => p.obs.filter((o) => poses[o.img]));
      function computeCost() {
        let cost = 0, cnt = 0;
        for (let pi = 0; pi < points.length; pi++) {
          for (const o of obsOf[pi]) {
            const f = feats[o.img], R = Rc.get(o.img), C = Cc.get(o.img);
            const d = [Xw[pi][0] - C[0], Xw[pi][1] - C[1], Xw[pi][2] - C[2]];
            const px = R[0] * d[0] + R[1] * d[1] + R[2] * d[2];
            const py = R[3] * d[0] + R[4] * d[1] + R[5] * d[2];
            const pz = R[6] * d[0] + R[7] * d[1] + R[8] * d[2];
            if (pz <= 1e-9) continue;
            const ex = f.f * px / pz + f.w / 2 - f.pts[o.kp][0];
            const ey = f.f * py / pz + f.h / 2 - f.pts[o.kp][1];
            if (ex * ex + ey * ey > 64) continue; // >8px: treat as outlier, exclude
            cost += ex * ex + ey * ey; cnt++;
          }
        }
        return { cost, cnt };
      }
      let { cost } = computeCost();
      for (let iter = 0; iter < iters; iter++) {
        // build normal equations
        const Hcc = Array.from({ length: nc }, () => new Float64Array(nc));
        const bc = new Float64Array(nc);
        const perPoint = []; // {Hpp: 3x3 array9, bp: [3], blocks: [{slot, Jc: 12 floats}], rhsP}
        for (let pi = 0; pi < points.length; pi++) {
          const Hpp = new Float64Array(9), bp = new Float64Array(3);
          const blocks = [];
          let used = 0;
          for (const o of obsOf[pi]) {
            const f = feats[o.img], R = Rc.get(o.img), C = Cc.get(o.img);
            const d = [Xw[pi][0] - C[0], Xw[pi][1] - C[1], Xw[pi][2] - C[2]];
            const px = R[0] * d[0] + R[1] * d[1] + R[2] * d[2];
            const py = R[3] * d[0] + R[4] * d[1] + R[5] * d[2];
            const pz = R[6] * d[0] + R[7] * d[1] + R[8] * d[2];
            if (pz <= 1e-9) continue;
            const ex = f.f * px / pz + f.w / 2 - f.pts[o.kp][0];
            const ey = f.f * py / pz + f.h / 2 - f.pts[o.kp][1];
            if (ex * ex + ey * ey > 64) continue;
            used++;
            const iz = 1 / pz, iz2 = iz * iz;
            const Jp2 = [f.f * iz, 0, -f.f * px * iz2, 0, f.f * iz, -f.f * py * iz2]; // 2x3 d(proj)/d(p)
            // d(proj)/dX = Jp2 * R
            const Jx = new Float64Array(6);
            for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++)
              Jx[r * 3 + c] = Jp2[r * 3] * R[c] + Jp2[r * 3 + 1] * R[3 + c] + Jp2[r * 3 + 2] * R[6 + c];
            for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
              Hpp[r * 3 + c] += Jx[c] * Jx[r * 3 + c] * 0 + Jx[r * 3 + 0] * 0; // placeholder replaced below
            }
            // (accumulate properly)
            for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
              Hpp[r * 3 + c] += Jx[0 * 3 + r] * Jx[0 * 3 + c] + Jx[1 * 3 + r] * Jx[1 * 3 + c];
            bp[0] -= Jx[0] * ex + Jx[3] * ey;
            bp[1] -= Jx[1] * ex + Jx[4] * ey;
            bp[2] -= Jx[2] * ex + Jx[5] * ey;
            if (o.img !== fixedCam) {
              // d(p)/d(w) = -[p]x ; d(p)/dC = -R
              const Jw = new Float64Array(6), JC = new Float64Array(6);
              const skew = [0, -pz, py, pz, 0, -px, -py, px, 0];
              for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) {
                Jw[r * 3 + c] = -(Jp2[r * 3] * skew[c] + Jp2[r * 3 + 1] * skew[3 + c] + Jp2[r * 3 + 2] * skew[6 + c]);
                JC[r * 3 + c] = -(Jp2[r * 3] * R[c] + Jp2[r * 3 + 1] * R[3 + c] + Jp2[r * 3 + 2] * R[6 + c]);
              }
              const Jc = new Float64Array(12); // 2x6: [Jw | JC]
              for (let r = 0; r < 2; r++) { for (let c = 0; c < 3; c++) { Jc[r * 6 + c] = Jw[r * 3 + c]; Jc[r * 6 + 3 + c] = JC[r * 3 + c]; } }
              const slot = camSlot.get(o.img);
              blocks.push({ slot, Jc, Jx, ex, ey });
              // Hcc += Jc^T Jc ; bc -= Jc^T e
              for (let r = 0; r < 6; r++) {
                bc[slot * 6 + r] -= Jc[r] * ex + Jc[6 + r] * ey;
                for (let c = 0; c < 6; c++) Hcc[slot * 6 + r][slot * 6 + c] += Jc[r] * Jc[c] + Jc[6 + r] * Jc[6 + c];
              }
            } else {
              blocks.push({ slot: -1, Jc: null, Jx, ex, ey });
            }
          }
          if (used >= 2) perPoint.push({ pi, Hpp, bp, blocks });
        }
        // Schur complement: S = Hcc - sum Hcp Hpp^-1 Hpc ; rhs = bc - sum Hcp Hpp^-1 bp
        const S = Hcc.map((row) => Float64Array.from(row));
        const rhs = Float64Array.from(bc);
        const pointInv = [];
        for (const pp of perPoint) {
          const HppA = [[pp.Hpp[0], pp.Hpp[1], pp.Hpp[2]], [pp.Hpp[3], pp.Hpp[4], pp.Hpp[5]], [pp.Hpp[6], pp.Hpp[7], pp.Hpp[8]]];
          for (let d = 0; d < 3; d++) HppA[d][d] += lambda * (HppA[d][d] + 1e-9);
          const inv = inv3(HppA);
          if (!inv) { pointInv.push(null); continue; }
          pointInv.push(inv);
          // accumulate per camera block: Hcp is 6x3 = sum over obs of Jc^T Jx
          const HcpBySlot = new Map();
          for (const bl of pp.blocks) {
            if (bl.slot < 0) continue;
            if (!HcpBySlot.has(bl.slot)) HcpBySlot.set(bl.slot, new Float64Array(18));
            const Hcp = HcpBySlot.get(bl.slot);
            for (let r = 0; r < 6; r++) for (let c = 0; c < 3; c++)
              Hcp[r * 3 + c] += bl.Jc[r] * bl.Jx[c] + bl.Jc[6 + r] * bl.Jx[3 + c];
          }
          const invBp = [0, 1, 2].map((r) => inv[r][0] * pp.bp[0] + inv[r][1] * pp.bp[1] + inv[r][2] * pp.bp[2]);
          for (const [slot, Hcp] of HcpBySlot) {
            // S -= Hcp * inv * Hcp^T ; rhs -= Hcp * invBp
            const tmp = new Float64Array(18); // Hcp(6x3) * inv(3x3) -> 6x3
            for (let r = 0; r < 6; r++) for (let c = 0; c < 3; c++)
              tmp[r * 3 + c] = Hcp[r * 3] * inv[0][c] + Hcp[r * 3 + 1] * inv[1][c] + Hcp[r * 3 + 2] * inv[2][c];
            for (let r = 0; r < 6; r++) {
              rhs[slot * 6 + r] -= tmp[r * 3] * pp.bp[0] * 0 + (Hcp[r * 3] * invBp[0] + Hcp[r * 3 + 1] * invBp[1] + Hcp[r * 3 + 2] * invBp[2]);
              for (let c = 0; c < 6; c++)
                S[slot * 6 + r][slot * 6 + c] -= tmp[r * 3] * Hcp[c * 3] + tmp[r * 3 + 1] * Hcp[c * 3 + 1] + tmp[r * 3 + 2] * Hcp[c * 3 + 2];
            }
          }
        }
        for (let d = 0; d < nc; d++) S[d][d] += lambda * (S[d][d] + 1e-9);
        const dc = solveDense(S.map((r) => Array.from(r)), Array.from(rhs));
        if (!dc || dc.some((v) => !isFinite(v))) { lambda *= 4; continue; }
        // apply camera updates (tentative copies)
        const Rc2 = new Map(Rc), Cc2 = new Map(Cc);
        for (const c of freeCams) {
          const slot = camSlot.get(c);
          const dw = [dc[slot * 6], dc[slot * 6 + 1], dc[slot * 6 + 2]];
          const dC = [dc[slot * 6 + 3], dc[slot * 6 + 4], dc[slot * 6 + 5]];
          Rc2.set(c, mm3(expSO3(dw), Rc.get(c)));
          const Cold = Cc.get(c);
          Cc2.set(c, [Cold[0] + dC[0], Cold[1] + dC[1], Cold[2] + dC[2]]);
        }
        // point updates: dX = inv * (bp - Hpc * dc)
        const Xw2 = Xw.map((x) => x.slice());
        for (let k = 0; k < perPoint.length; k++) {
          const pp = perPoint[k], inv = pointInv[k];
          if (!inv) continue;
          const corr = [pp.bp[0], pp.bp[1], pp.bp[2]];
          for (const bl of pp.blocks) {
            if (bl.slot < 0) continue;
            for (let r = 0; r < 3; r++) {
              let ssum = 0;
              for (let c = 0; c < 6; c++) ssum += (bl.Jc[c] * 0 + (c < 3 ? bl.Jx[r] * 0 : 0)); // placeholder
              void ssum;
            }
          }
          // Hpc * dc needs Jx^T Jc * dc summed over obs of this point & camera:
          const HpcDc = [0, 0, 0];
          for (const bl of pp.blocks) {
            if (bl.slot < 0) continue;
            const dcs = [dc[bl.slot * 6], dc[bl.slot * 6 + 1], dc[bl.slot * 6 + 2], dc[bl.slot * 6 + 3], dc[bl.slot * 6 + 4], dc[bl.slot * 6 + 5]];
            // (Jc^T applied to residual space) -> we need Hpc (3x6) = Jx^T * Jc
            for (let r = 0; r < 3; r++) {
              let ssum = 0;
              for (let c = 0; c < 6; c++) {
                const JcElem = c < 3 ? 0 : 0;
                void JcElem;
                // Jx rows: [Jx[0..2]] (row0), [Jx[3..5]] (row1); Jc rows: [Jc[0..5]], [Jc[6..11]]
                ssum += (bl.Jx[r] * bl.Jc[c] + bl.Jx[3 + r] * bl.Jc[6 + c]) * dcs[c];
              }
              HpcDc[r] += ssum;
            }
          }
          const rhsP = [corr[0] - HpcDc[0], corr[1] - HpcDc[1], corr[2] - HpcDc[2]];
          const dX = [0, 1, 2].map((r) => inv[r][0] * rhsP[0] + inv[r][1] * rhsP[1] + inv[r][2] * rhsP[2]);
          Xw2[pp.pi] = [Xw[pp.pi][0] + dX[0], Xw[pp.pi][1] + dX[1], Xw[pp.pi][2] + dX[2]];
        }
        // evaluate tentative cost
        const RcSave = new Map(Rc), CcSave = new Map(Cc), XwSave = Xw.map((x) => x.slice());
        for (const c of camIdx) { Rc.set(c, Rc2.get(c)); Cc.set(c, Cc2.get(c)); }
        for (let pi = 0; pi < points.length; pi++) Xw[pi] = Xw2[pi];
        const nc2 = computeCost();
        if (nc2.cost < cost) {
          cost = nc2.cost;
          lambda = Math.max(lambda / 3, 1e-7);
        } else {
          for (const c of camIdx) { Rc.set(c, RcSave.get(c)); Cc.set(c, CcSave.get(c)); }
          for (let pi = 0; pi < points.length; pi++) Xw[pi] = XwSave[pi];
          lambda = Math.min(lambda * 4, 1e2);
        }
      }
      // write back
      for (const c of camIdx) {
        const R = Rc.get(c), C = Cc.get(c);
        poses[c] = { R, t: [-(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]), -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]), -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2])] };
      }
      for (let pi = 0; pi < points.length; pi++) points[pi].xyz = Xw[pi];
      return cost;
    }
    function inv3(A) {
      const [a, b, c] = A[0], [d, e, f] = A[1], [g, h, i] = A[2];
      const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
      if (Math.abs(det) < 1e-18) return null;
      const s = 1 / det;
      return [
        [(e * i - f * h) * s, (c * h - b * i) * s, (b * f - c * e) * s],
        [(f * g - d * i) * s, (a * i - c * g) * s, (c * d - a * f) * s],
        [(d * h - e * g) * s, (b * g - a * h) * s, (a * e - b * d) * s],
      ];
    }

    // ---- helpers for registration ----
    function corrMapFor(j) {
      const corr = new Map();
      for (let i = 0; i < n; i++) {
        if (!poses[i]) continue;
        const ms = oriented(i, j);
        if (!ms) continue;
        for (const m of ms) {
          const pIdx = kpPoint[i][m.q];
          if (pIdx >= 0 && !corr.has(m.t)) corr.set(m.t, pIdx);
        }
      }
      return corr;
    }
    function tryPnP(j, minInl) {
      const need = minInl || MIN_PNP_INLIERS;
      const corr = corrMapFor(j);
      if (corr.size < need + 6) return 0;
      const corrKeys = [...corr.keys()];
      const obj = [], imgPts = [];
      for (const kp of corrKeys) { obj.push(points[corr.get(kp)].xyz); imgPts.push(feats[j].pts[kp]); }
      const fj = feats[j].f;
      const K = cv.matFromArray(3, 3, cv.CV_64F, [fj, 0, feats[j].w / 2, 0, fj, feats[j].h / 2, 0, 0, 1]);
      const dist = cv.Mat.zeros(4, 1, cv.CV_64F);
      const flat3 = []; for (const p of obj) flat3.push(p[0], p[1], p[2]);
      const flat2 = []; for (const p of imgPts) flat2.push(p[0], p[1]);
      const objM = cv.matFromArray(obj.length, 1, cv.CV_64FC3, flat3);
      const imgM = cv.matFromArray(imgPts.length, 1, cv.CV_64FC2, flat2);
      const rvec = new cv.Mat(), tvec = new cv.Mat(), inliers = new cv.Mat();
      let ok = false;
      try {
        ok = cv.solvePnPRansac(objM, imgM, K, dist, rvec, tvec, false, 120, 8.0, 0.99, inliers, cv.SOLVEPNP_EPNP);
        if (!ok || inliers.rows < need)
          ok = cv.solvePnPRansac(objM, imgM, K, dist, rvec, tvec, false, 200, 8.0, 0.99, inliers, cv.SOLVEPNP_ITERATIVE);
      } catch (e) { ok = false; }
      const inlCount = ok ? inliers.rows : 0;
      let result = 0;
      if (ok && inlCount >= need) {
        const R = new cv.Mat();
        cv.Rodrigues(rvec, R);
        poses[j] = { R: Array.from(R.data64F), t: Array.from(tvec.data64F) };
        R.delete();
        const inl = inliers.data32S;
        for (let k = 0; k < inl.length; k++) {
          const kpIdx = corrKeys[inl[k]];
          if (kpPoint[j][kpIdx] < 0) attach(j, kpIdx, corr.get(kpIdx));
        }
        result = inlCount;
      }
      K.delete(); dist.delete(); objM.delete(); imgM.delete(); rvec.delete(); tvec.delete(); inliers.delete();
      return result;
    }
    function resetState() {
      poses.fill(null);
      for (const arr of kpPoint) arr.fill(-1);
      points.length = 0;
    }
    const pairCount = (a, b) => { const r = pairMatches.get(pairKey(a, b)); return r ? r.matches.length : 0; };

    const sortedPairs = [...pairMatches.values()].sort((x, y) => y.matches.length - x.matches.length);

    // ---- init pair (validated empirically: a 3rd image must PnP-register) ----
    // 焦距若不準 (無 EXIF 時是猜的), 整個幾何會系統性歪掉; 用 init 驗證當裁判掃幾個焦距倍率.
    let initDone = false;
    for (const fScale of [1.0, 0.85, 1.18]) {
      for (const fe of feats) fe.f = fe.fBase * fScale;
      for (const rec of sortedPairs.slice(0, 6)) {
      resetState();
      const { i, j, matches } = rec;
      const fi = feats[i].f, fj = feats[j].f;
      const est = estimateEssential(
        matches.map((m) => feats[i].pts[m.q]), matches.map((m) => feats[j].pts[m.t]),
        fi, [feats[i].w / 2, feats[i].h / 2], fj, [feats[j].w / 2, feats[j].h / 2]);
      if (!est || est.inliers.length < MIN_INIT_INLIERS) continue;
      const dec = decomposeEssential(est.E, est.b1, est.b2, est.inliers, fi, [feats[i].w / 2, feats[i].h / 2], fj, [feats[j].w / 2, feats[j].h / 2]);
      if (!dec || dec.pos < MIN_INIT_INLIERS * 0.6) continue;
      poses[i] = { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] };
      poses[j] = { R: dec.R, t: dec.t };
      triangulatePair(i, j);
      if (points.length < 150) continue;
      const others = feats.map((_, idx) => idx).filter((x) => x !== i && x !== j)
        .sort((a, b) => (pairCount(b, i) + pairCount(b, j)) - (pairCount(a, i) + pairCount(a, j)));
      let validated = false;
      for (const cand of others.slice(0, 3)) {
        if (tryPnP(cand) > 0) {
          triangulatePair(i, cand);
          triangulatePair(j, cand);
          validated = true; break;
        }
      }
      if (!validated) continue;
      initDone = true;
      break;
      }
      if (initDone) break;
    }
    if (!initDone) throw new Error("找不到夠穩的起始影像對 — 試著讓相鄰照片重疊更多、或增加照片數量");
    bundleAdjust(15);
    onProgress && onProgress("init", poses.filter(Boolean).length, n);
    await sleep();

    // ---- incremental PnP registration ----
    const failed = new Set();
    let guard = 0;
    for (;;) {
      const remaining = feats.map((_, idx) => idx).filter((idx) => !poses[idx] && !failed.has(idx));
      if (!remaining.length || guard++ > 4 * n) break;
      let bestJ = -1, bestSize = -1;
      for (const j of remaining) {
        const sz = corrMapFor(j).size;
        if (sz > bestSize) { bestSize = sz; bestJ = j; }
      }
      if (bestJ < 0 || bestSize < MIN_PNP_INLIERS + 6) break;
      if (tryPnP(bestJ) > 0) {
        for (let i = 0; i < n; i++) if (poses[i] && i !== bestJ) triangulatePair(i, bestJ);
        bundleAdjust(8);
        failed.clear();
        onProgress && onProgress("register", poses.filter(Boolean).length, n);
      } else {
        failed.add(bestJ);
      }
      await sleep();
    }

    // ---- mop-up: one relaxed pass for images that almost made it ----
    if (poses.filter(Boolean).length >= 3 && poses.filter(Boolean).length < n) {
      const failed2 = new Set();
      let g2 = 0;
      for (;;) {
        const rem2 = feats.map((_, idx) => idx).filter((idx) => !poses[idx] && !failed2.has(idx));
        if (!rem2.length || g2++ > 2 * n) break;
        let bJ = -1, bS = -1;
        for (const j of rem2) { const sz = corrMapFor(j).size; if (sz > bS) { bS = sz; bJ = j; } }
        if (bJ < 0 || bS < 22) break;
        if (tryPnP(bJ, 16) > 0) {
          for (let i = 0; i < n; i++) if (poses[i] && i !== bJ) triangulatePair(i, bJ);
          bundleAdjust(8);
          onProgress && onProgress("register", poses.filter(Boolean).length, n);
        } else failed2.add(bJ);
        await sleep();
      }
    }

    bundleAdjust(12);
    const registered = poses.filter(Boolean).length;
    const kept = [];
    for (const p of points) {
      if (p.obs.length < 2) continue;
      let err = 0, cnt = 0, bad = false;
      for (const o of p.obs) {
        const pose = poses[o.img]; if (!pose) continue;
        const f = feats[o.img];
        const pr = projectPx(f.f, f.w / 2, f.h / 2, pose.R, pose.t, p.xyz);
        if (!pr) { bad = true; break; }
        err += Math.hypot(pr[0] - f.pts[o.kp][0], pr[1] - f.pts[o.kp][1]); cnt++;
      }
      if (!bad && cnt && err / cnt <= MAX_REPROJ_PX) kept.push(p);
    }
    const m = kept.length;
    const xyz = new Float32Array(m * 3), rgb = new Uint8Array(m * 3);
    for (let i = 0; i < m; i++) {
      xyz[i * 3] = kept[i].xyz[0]; xyz[i * 3 + 1] = kept[i].xyz[1]; xyz[i * 3 + 2] = kept[i].xyz[2];
      const c = kept[i].rgbCnt ? kept[i].rgbSum.map((v) => v / kept[i].rgbCnt) : [200, 200, 200];
      rgb[i * 3] = c[0]; rgb[i * 3 + 1] = c[1]; rgb[i * 3 + 2] = c[2];
    }
    for (const f of feats) f.desc.delete();
    return { xyz, rgb, registered, total: n, rawPoints: points.length };
  }

  /* ---------------- 點雲 -> Gaussian splat (.splat, 32B/個) ---------------- */
  function pointsToSplat(xyz, rgb, maxPoints) {
    let n = xyz.length / 3;
    let idx = [...Array(n).keys()];
    if (maxPoints && n > maxPoints) {
      const step = n / maxPoints;
      idx = [];
      for (let i = 0; i < n && idx.length < maxPoints; i += step) idx.push(Math.floor(i));
      n = idx.length;
    }
    let P = idx.map((i) => [xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]]);
    let SRC = idx.slice();
    const med = [0, 1, 2].map((ax) => {
      const col = P.map((p) => p[ax]).sort((a, b) => a - b);
      return col[Math.floor(col.length / 2)];
    });
    for (const p of P) { p[0] -= med[0]; p[1] -= med[1]; p[2] -= med[2]; p[1] *= -1; p[2] *= -1; }
    { // PCA 正規化方向: SfM 的世界座標是任意的, 不對齊的話模型常以奇怪的角度/正對平面呈現.
      // 選最接近鉛直的主軸當 +Y, 次大水平主軸當 X (最長邊橫向), 法線方向當 Z.
      const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (const p of P) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) cov[a][b] += p[a] * p[b];
      const eig = jacobiEigen(cov);
      const axes = [eig.vectors[0], eig.vectors[1], eig.vectors[2]];
      let upIdx = 0;
      for (let i = 1; i < 3; i++) if (Math.abs(axes[i][1]) > Math.abs(axes[upIdx][1])) upIdx = i;
      let Y = axes[upIdx].slice();
      if (Y[1] < 0) Y = Y.map((v) => -v);
      const rest = axes.filter((_, i) => i !== upIdx);
      let X = rest[0].slice();
      let Z = [X[1] * Y[2] - X[2] * Y[1], X[2] * Y[0] - X[0] * Y[2], X[0] * Y[1] - X[1] * Y[0]];
      if (Z[0] * rest[1][0] + Z[1] * rest[1][1] + Z[2] * rest[1][2] < 0) { X = X.map((v) => -v); Z = Z.map((v) => -v); }
      for (const p of P) {
        const q = [p[0] * X[0] + p[1] * X[1] + p[2] * X[2],
                   p[0] * Y[0] + p[1] * Y[1] + p[2] * Y[2],
                   p[0] * Z[0] + p[1] * Z[1] + p[2] * Z[2]];
        p[0] = q[0]; p[1] = q[1]; p[2] = q[2];
      }
    }
    const dists = P.map((p) => Math.hypot(p[0], p[1], p[2])).sort((a, b) => a - b);
    const r95 = dists[Math.floor(dists.length * 0.95)] || 1;
    { // trim far outliers (stray points on near-parallel rays)
      const keep = [];
      for (let i = 0; i < P.length; i++) if (Math.hypot(P[i][0], P[i][1], P[i][2]) <= r95 * 1.6) keep.push(i);
      if (keep.length >= 50 && keep.length < P.length) {
        P = keep.map((i) => P[i]); SRC = keep.map((i) => SRC[i]); n = P.length;
      }
    }
    const s = 4.0 / r95;
    const scales = new Float32Array(n);
    for (let a = 0; a < n; a++) {
      const ds = [];
      for (let b = 0; b < n; b++) {
        if (a === b) continue;
        const dx = P[a][0] - P[b][0], dy = P[a][1] - P[b][1], dz = P[a][2] - P[b][2];
        ds.push(dx * dx + dy * dy + dz * dz);
      }
      ds.sort((x, y) => x - y);
      const k = Math.min(3, ds.length);
      let mean = 0; for (let q = 0; q < k; q++) mean += Math.sqrt(ds[q]);
      scales[a] = (mean / k) * s;
    }
    const buf = new ArrayBuffer(n * 32);
    const dv = new DataView(buf);
    const u8 = new Uint8Array(buf);
    for (let i = 0; i < n; i++) {
      const o = i * 32, src = SRC[i];
      dv.setFloat32(o + 0, P[i][0] * s, true);
      dv.setFloat32(o + 4, P[i][1] * s, true);
      dv.setFloat32(o + 8, P[i][2] * s, true);
      const sc = Math.min(0.28, Math.max(0.012, scales[i] * 1.15));
      dv.setFloat32(o + 12, sc, true); dv.setFloat32(o + 16, sc, true); dv.setFloat32(o + 20, sc, true);
      u8[o + 24] = rgb[src * 3]; u8[o + 25] = rgb[src * 3 + 1]; u8[o + 26] = rgb[src * 3 + 2];
      u8[o + 27] = 205;
      u8[o + 28] = 255; u8[o + 29] = 128; u8[o + 30] = 128; u8[o + 31] = 128;
    }
    return buf;
  }

  global.ClimbSfM = { runSfM, pointsToSplat };
})(typeof window !== "undefined" ? window : globalThis);
