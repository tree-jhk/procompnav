/* You vs ProCompNav — static mini-game. Vanilla JS, no build step.
 * Data: data/manifest.json -> data/rounds/<id>.json (see DATA_SPEC.md). All paths are relative.
 */
// ===== Share link base =====================================================================
// Canonical URL of the game page used in shared links ("<base>#r=<round_id>").
// Empty string -> use the current page URL (fine when the game is opened directly).
// Set it to the public page when the game is embedded in an iframe, e.g.
//   const SHARE_BASE_URL = "https://tree-jhk.github.io/procompnav/game/";
// Shared links point at pre-generated preview pages s/<round>-y<N>.html (tools/make_share_previews.py)
// when s/index.json lists them, otherwise at <base>#r=<round>.
const SHARE_BASE_URL = "";
// ===========================================================================================

(function () {
  'use strict';

  // ---------- tiny helpers ----------
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var el = function (tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === 'class') n.className = attrs[k];
      else if (k === 'text') n.textContent = attrs[k];
      else if (k === 'html') n.innerHTML = attrs[k];
      else if (k.slice(0, 2) === 'on') n.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== null && attrs[k] !== undefined && attrs[k] !== false) n.setAttribute(k, attrs[k] === true ? '' : attrs[k]);
    }
    (kids || []).forEach(function (c) { if (c != null) n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return n;
  };
  var reduceMQ = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };
  var wait = function (ms) { return new Promise(function (r) { setTimeout(r, reduceMQ.matches ? Math.min(ms, 80) : ms); }); };
  var plural = function (n, w) { return n + ' ' + w + (n === 1 ? '' : /(s|x|z|ch|sh)$/.test(w) ? 'es' : 's'); };
  var LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  var CAT_ORDER = ['cabinet', 'bed', 'table', 'desk', 'sofa', 'chair'];
  var capital = function (s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; };

  // ---------- local storage (best effort) ----------
  var STORE_KEY = 'procompnav-game-v1';
  function loadStore() {
    try { var v = JSON.parse(window.localStorage.getItem(STORE_KEY)); return v && typeof v === 'object' ? v : {}; }
    catch (e) { return {}; }
  }
  function saveStore() { try { window.localStorage.setItem(STORE_KEY, JSON.stringify(S)); } catch (e) { /* private mode etc. */ } }
  var S = loadStore();
  S.tally = S.tally || { played: 0, win: 0, tie: 0, loss: 0 };
  S.done = S.done || {};

  // ---------- theme ----------
  function applyTheme() { if (S.theme) document.documentElement.setAttribute('data-theme', S.theme); }
  applyTheme();
  $('#theme-btn').addEventListener('click', function () {
    var cur = document.documentElement.getAttribute('data-theme');
    if (!cur) cur = (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
    S.theme = cur === 'dark' ? 'light' : 'dark';
    applyTheme(); saveStore();
  });

  // In an iframe: offer a "Full screen" link that opens the game on its own.
  try { if (window.self !== window.top) $('#open-full').hidden = false; } catch (e) { $('#open-full').hidden = false; }

  // ---------- blocky SVG art ----------
  function rects(list) {
    return list.map(function (r) { return '<rect x="' + r[0] + '" y="' + r[1] + '" width="' + r[2] + '" height="' + r[3] + '" fill="' + r[4] + '"/>'; }).join('');
  }
  function svg16(inner, label) {
    return '<svg viewBox="0 0 16 16" shape-rendering="crispEdges" ' + (label ? 'role="img" aria-label="' + label + '"' : 'aria-hidden="true"') + '>' + inner + '</svg>';
  }
  function robotRects(body, dark, eye) {
    return [
      [7, 0, 2, 1, eye], [7.5, 1, 1, 2, '#1b241d'],
      [2, 3, 12, 11, '#1b241d'], [3, 4, 10, 9, body], [3, 4, 10, 1, 'rgba(255,255,255,.35)'],
      [4, 6, 8, 4, '#1b241d'], [5, 7, 2, 2, eye], [9, 7, 2, 2, eye],
      [6, 11, 4, 1, dark], [1, 7, 1, 3, '#1b241d'], [14, 7, 1, 3, '#1b241d'],
      [4, 14, 8, 2, '#1b241d'], [5, 14, 6, 1, dark]
    ];
  }
  function robotSVG(body, dark, eye) { return svg16(rects(robotRects(body, dark, eye))); }
  function humanSVG(mood) {
    var skin = '#e9b98c', hair = '#4a2f1b', ink = '#1b241d';
    var base = [
      [2, 1, 12, 14, ink], [3, 2, 10, 12, skin], [3, 2, 10, 3, hair], [3, 5, 1, 3, hair], [12, 5, 1, 2, hair],
      [5, 7, 2, 2, ink], [9, 7, 2, 2, ink], [5, 7, 1, 1, '#fff'], [9, 7, 1, 1, '#fff'], [3, 14, 10, 1, '#2f7fb3']
    ];
    var face;
    if (mood === 'mad') face = [[4, 6, 3, 1, ink], [9, 6, 3, 1, ink], [6, 12, 4, 1, ink], [5, 11, 1, 1, ink], [10, 11, 1, 1, ink], [3, 10, 2, 1, '#e0675c'], [11, 10, 2, 1, '#e0675c']];
    else if (mood === 'meh') face = [[6, 11, 4, 1, ink], [4, 6, 2, 1, hair], [10, 6, 2, 1, hair]];
    else face = [[5, 10, 1, 1, ink], [10, 10, 1, 1, ink], [6, 11, 4, 1, ink], [4, 10, 1, 1, '#f09a8a'], [11, 10, 1, 1, '#f09a8a']];
    return svg16(rects(base.concat(face)));
  }
  var AV = {
    you: function () { return robotSVG('#46a035', '#2c6e20', '#b8f59a'); },
    bot: function () { return robotSVG('#f0ad1f', '#9a6a00', '#fff2b8'); },
    human: function (mood) { return humanSVG(mood); }
  };
  function paintAvatars(root) {
    (root || document).querySelectorAll('[data-avatar]').forEach(function (n) {
      var k = n.getAttribute('data-avatar');
      if (AV[k]) n.innerHTML = AV[k](n.getAttribute('data-mood'));
    });
  }
  function setMood(mood) {
    ['#human-avatar', '#msg-human-av'].forEach(function (s) { var n = $(s); n.setAttribute('data-mood', mood); n.innerHTML = AV.human(mood); });
  }
  var HEART = rects([[1, 0, 2, 1, 'c'], [4, 0, 2, 1, 'c'], [0, 1, 7, 2, 'c'], [1, 3, 5, 1, 'c'], [2, 4, 3, 1, 'c'], [3, 5, 1, 1, 'c']]).replace(/ fill="c"/g, '');
  var CAT_ICON = {
    cabinet: [[2, 2, 12, 12, 'E'], [3, 3, 4, 10, 'P'], [9, 3, 4, 10, 'P'], [6, 7, 1, 2, 'E'], [9, 7, 1, 2, 'E'], [3, 14, 2, 1, 'E'], [11, 14, 2, 1, 'E']],
    bed: [[1, 4, 2, 10, 'E'], [3, 8, 12, 4, 'E'], [3, 6, 4, 2, 'P'], [4, 9, 10, 2, 'P'], [13, 12, 2, 2, 'E'], [3, 12, 1, 2, 'E']],
    table: [[1, 5, 14, 3, 'E'], [2, 6, 12, 1, 'P'], [2, 8, 2, 6, 'E'], [12, 8, 2, 6, 'E']],
    desk: [[1, 4, 14, 3, 'E'], [2, 5, 12, 1, 'P'], [9, 7, 5, 7, 'E'], [10, 8, 3, 2, 'P'], [10, 11, 3, 2, 'P'], [2, 7, 2, 7, 'E']],
    sofa: [[1, 6, 3, 7, 'E'], [12, 6, 3, 7, 'E'], [3, 4, 10, 5, 'E'], [4, 5, 8, 3, 'P'], [4, 9, 8, 3, 'E'], [5, 10, 6, 1, 'P']],
    random: [[3, 3, 10, 10, 'E'], [4, 4, 8, 8, 'P'], [5, 5, 2, 2, 'E'], [9, 9, 2, 2, 'E'], [7, 7, 2, 2, 'E']],
    other: [[3, 3, 10, 10, 'E'], [4, 4, 8, 3, 'G'], [4, 7, 8, 5, 'D']]
  };
  function catIcon(cat) {
    var spec = CAT_ICON[cat] || CAT_ICON.other;
    var map = { E: 'currentColor', P: 'var(--surface-2)', G: 'var(--grass)', D: 'var(--dirt)' };
    if (cat === 'random') map.P = 'var(--surface)';
    return svg16(rects(spec.map(function (r) { return [r[0], r[1], r[2], r[3], map[r[4]]]; })));
  }
  var ICON_VIEWS = '<svg viewBox="0 0 16 16" aria-hidden="true" shape-rendering="crispEdges"><path d="M1 1h6v6H1zM9 1h6v6H9zM1 9h6v6H1zM9 9h6v6H9z"/></svg>';
  var ICON_X = '<svg class="x" viewBox="0 0 16 16" aria-hidden="true" shape-rendering="crispEdges"><path d="M2 2h3v2H2zM4 4h2v2H4zM6 6h4v4H6zM10 4h2v2h-2zM11 2h3v2h-3zM4 10h2v2H4zM2 12h3v2H2zM10 10h2v2h-2zM11 12h3v2h-3z"/></svg>';

  // ---------- shuffle (fresh every play: card layout, labels and question order all change) ----------
  function freshSeed() { return ((Math.random() * 4294967296) >>> 0) || 1; }
  function shuffled(arr, seed) {
    var a = arr.slice(), x = seed || 1;
    var rnd = function () { x ^= x << 13; x ^= x >>> 17; x ^= x << 5; return ((x >>> 0) % 100000) / 100000; };
    for (var i = a.length - 1; i > 0; i--) { var j = Math.floor(rnd() * (i + 1)); var t = a[i]; a[i] = a[j]; a[j] = t; }
    return a;
  }

  // ---------- toast + announcements ----------
  var toastTimer = null;
  function toast(big, small, kind, ms) {
    var t = $('#toast');
    $('#toast-big').textContent = big;
    $('#toast-small').textContent = small || '';
    $('#toast-small').hidden = !small;
    t.className = 'toast show ' + (kind || '');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.className = 'toast ' + (kind || ''); }, ms || 2200);
  }
  function announce(msg) { var n = $('#sr-live'); n.textContent = ''; setTimeout(function () { n.textContent = msg; }, 30); }

  // ---------- screens ----------
  function show(id) {
    ['screen-title', 'screen-round', 'screen-result'].forEach(function (s) { $('#' + s).hidden = s !== id; });
    clearTimeout(toastTimer); $('#toast').className = 'toast';
    window.scrollTo(0, 0);
  }
  function setHash(h) {
    try { history.replaceState(null, '', h ? '#' + h : location.pathname + location.search); } catch (e) { /* sandboxed iframe */ }
  }

  // ---------- data ----------
  var manifest = null;
  var roundCache = {};
  function fetchJSON(url) {
    return fetch(url, { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) throw new Error(r.status + ' ' + url);
      return r.json();
    });
  }
  function loadRound(id) {
    if (roundCache[id]) return Promise.resolve(roundCache[id]);
    return fetchJSON(manifest.rounds[id]).then(function (r) { roundCache[id] = r; return r; });
  }
  function realRounds() {
    var out = [];
    Object.keys(manifest.categories || {}).forEach(function (c) { out = out.concat(manifest.categories[c]); });
    return out;
  }
  function sortedCats() {
    return Object.keys(manifest.categories || {}).filter(function (c) { return manifest.categories[c].length; }).sort(function (a, b) {
      var ia = CAT_ORDER.indexOf(a), ib = CAT_ORDER.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
    });
  }
  function info(id) { return (manifest.info && manifest.info[id]) || {}; }
  function pickFrom(ids, avoid) {
    var pool = ids.filter(function (id) { return !S.done[id] && id !== avoid; });
    if (!pool.length) pool = ids.filter(function (id) { return id !== avoid; });
    if (!pool.length) pool = ids;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  // ---------- title screen ----------
  var openCat = null;
  function renderTitle() {
    paintAvatars($('#screen-title'));
    var demo = (manifest.demo || [])[0];
    var wb = $('#warmup-btn');
    wb.hidden = !demo;
    if (demo) {
      var di = info(demo);
      $('#warmup-sub').textContent = (di.title ? di.title.replace(/^Warm-up:\s*/i, '') : 'Practice round') + (S.done[demo] ? ' · played' : '');
      wb.onclick = function () { startRound(demo); };
    }
    var cats = sortedCats();
    $('#picker').hidden = !cats.length;
    var grid = $('#cat-grid');
    grid.innerHTML = '';
    cats.forEach(function (c) {
      var ids = manifest.categories[c];
      var ns = ids.map(function (id) { return info(id).n_cards || 0; });
      var lo = Math.min.apply(null, ns), hi = Math.max.apply(null, ns);
      var done = ids.filter(function (id) { return S.done[id]; }).length;
      var b = el('button', { type: 'button', class: 'cat', 'aria-pressed': openCat === c ? 'true' : 'false', 'aria-controls': 'round-pills' }, []);
      b.innerHTML = catIcon(c);
      b.appendChild(el('span', { class: 'cat-name', text: capital(c) }));
      b.appendChild(el('span', { class: 'cat-meta', text: plural(ids.length, 'round') + ' · ' + (lo === hi ? lo : lo + '–' + hi) + ' cards' + (done ? ' · ' + done + ' played' : '') }));
      b.addEventListener('click', function () { openCat = openCat === c ? null : c; renderTitle(); if (openCat) { var f = $('#round-pills .pill'); if (f) f.focus(); } });
      grid.appendChild(b);
    });
    if (cats.length) {
      var all = realRounds();
      var rb = el('button', { type: 'button', class: 'cat random' });
      rb.innerHTML = catIcon('random');
      rb.appendChild(el('span', { class: 'cat-name', text: 'Random' }));
      rb.appendChild(el('span', { class: 'cat-meta', text: 'any of ' + plural(all.length, 'round') }));
      rb.addEventListener('click', function () { startRound(pickFrom(all)); });
      grid.appendChild(rb);
    }
    var pills = $('#round-pills');
    pills.innerHTML = '';
    pills.hidden = !openCat;
    if (openCat) {
      pills.appendChild(el('p', { text: capital(openCat) + ' rounds' }));
      manifest.categories[openCat].forEach(function (id, i) {
        var inf = info(id), res = S.done[id];
        var p = el('button', { type: 'button', class: 'pill', 'aria-label': 'Round ' + (i + 1) + ': ' + inf.n_cards + ' cards' + (res ? ', ' + resWord(res) : '') }, [
          el('span', { text: inf.n_cards + ' cards' }),
          res ? el('span', { class: 'res ' + res, text: resWord(res) }) : null
        ]);
        p.addEventListener('click', function () { startRound(id); });
        pills.appendChild(p);
      });
      var sp = el('button', { type: 'button', class: 'pill', text: 'Surprise me' });
      sp.addEventListener('click', function () { startRound(pickFrom(manifest.categories[openCat])); });
      pills.appendChild(sp);
    }
    renderTally($('#tally-title'), true);
  }
  function resWord(r) { return r === 'win' ? 'won' : r === 'tie' ? 'tie' : 'lost'; }
  function renderTally(node, withReset) {
    var t = S.tally;
    node.innerHTML = '';
    if (!t.played) { node.textContent = withReset ? 'Your record vs ProCompNav will show up here.' : ''; return; }
    node.appendChild(document.createTextNode('Your record vs ProCompNav: '));
    // W in green, T as is, L in red
    node.appendChild(el('b', { 'aria-label': t.win + ' wins, ' + t.tie + ' ties, ' + t.loss + ' losses' }, [
      el('span', { class: 't-w', text: t.win + 'W' }), document.createTextNode(' ' + t.tie + 'T '), el('span', { class: 't-l', text: t.loss + 'L' })
    ]));
    node.appendChild(document.createTextNode(' · ' + plural(t.played, 'round')));
    if (withReset) {
      node.appendChild(document.createTextNode(' · '));
      node.appendChild(el('button', { type: 'button', class: 'linkish', text: 'reset', onclick: function () {
        S.tally = { played: 0, win: 0, tie: 0, loss: 0 }; S.done = {}; saveStore(); renderTitle();
      } }));
    }
  }
  function goHome() { if (G) G.dead = true; G = null; setHash(''); renderTitle(); show('screen-title'); $('#title-h').focus({ preventScroll: true }); }
  $('#brand').addEventListener('click', goHome);
  $('#home-btn').addEventListener('click', goHome);

  // ---------- game state ----------
  var G = null;

  function startRound(id) {
    if (!manifest.rounds[id]) return;
    return loadRound(id).then(function (r) { setupRound(id, r); }).catch(function (err) {
      showLoadError('Could not load this round (' + err.message + ').');
    });
  }

  function setupRound(id, r) {
    if (G) G.dead = true;
    // New layout every play; labels A, B, C... follow this play's on-screen order.
    var cands = shuffled(r.candidates, freshSeed()).map(function (c, i) {
      return { id: c.id, letter: LETTERS[i] || String(i + 1), img: c.img, views: (c.views && c.views.length) ? c.views : [c.img] };
    });
    var byId = {};
    cands.forEach(function (c) { byId[c.id] = c; });
    var qmap = {};
    r.questions.forEach(function (q) { qmap[q.id] = q; });
    var robotQ = r.procompnav.steps.length;
    var hearts = Math.max(5, robotQ + 3);
    G = {
      id: id, r: r, cands: cands, byId: byId, qmap: qmap, robotQ: robotQ,
      alive: cands.map(function (c) { return c.id; }), asked: [], outBy: {},
      hearts: hearts, maxHearts: hearts, busy: false, over: false
    };
    setHash('r=' + encodeURIComponent(id));

    var cat = r.category || 'object';
    $('#query-text').textContent = r.query || ('Find the ' + cat + '.');
    $('#round-h').textContent = 'Round: ' + (r.query || cat);
    $('#rival-q').textContent = '?';
    $('#rival').setAttribute('aria-label', 'ProCompNav\'s question count is hidden until the end');
    $('#board-hint').innerHTML = '';
    $('#board-hint').append(plural(cands.length, cat) + ' here, only one is right. Tap a photo to see every view. Tap ', el('b', { text: 'Go here' }), ' when you’re sure.');
    $('#bank-note').textContent = 'Every question costs one heart. After each answer, questions that can no longer split the cards left disappear.';
    $('#asked-log').innerHTML = '';
    $('#asked-log').hidden = true;
    $('#msg-you').hidden = true; $('#msg-human').hidden = true; $('#msg-human-txt').classList.remove('long');
    setMood('happy');

    // cards
    var ul = $('#cards');
    ul.innerHTML = '';
    ul.classList.toggle('dense', cands.length > 6);
    cands.forEach(function (c) { ul.appendChild(cardEl(c)); });

    // bank: fresh random order every play, so the robot's questions are never listed first
    var bank = $('#bank');
    bank.innerHTML = '';
    bank.classList.remove('locked');
    shuffled(r.questions, freshSeed()).forEach(function (q) {
      var b = el('button', { type: 'button', class: 'chip', 'data-q': q.id }, [el('span', { class: 'chip-txt', text: q.text })]);
      b.addEventListener('click', function () { ask(q.id); });
      bank.appendChild(b);
    });

    renderStatus();
    paintAvatars($('#screen-round'));
    setMood('happy');
    show('screen-round');
    startClock(G);
    $('#round-h').focus({ preventScroll: true });
    announce('New round. ' + (r.query || '') + ' ' + plural(cands.length, 'card') + '. ' + plural(G.maxHearts, 'heart') + ' of patience.');
  }

  function cardEl(c) {
    var nv = c.views.length;
    var photo = el('button', { type: 'button', class: 'photo', 'aria-label': 'Card ' + c.letter + ': see ' + (nv > 1 ? 'all ' + nv + ' views' : 'larger') }, [
      el('img', { src: c.img, alt: 'Candidate ' + c.letter, decoding: 'async', width: 480, height: 360 }),
      el('span', { class: 'tag', text: c.letter, 'aria-hidden': 'true' }),
      nv > 1 ? el('span', { class: 'nviews', html: ICON_VIEWS + nv, 'aria-hidden': 'true' }) : null
    ]);
    photo.addEventListener('click', function () { openLightbox(c.id, 0); });
    var go = el('button', { type: 'button', class: 'go', text: 'Go here', 'aria-label': 'Go to card ' + c.letter });
    go.addEventListener('click', function () { guess(c.id); });
    var front = el('div', { class: 'face front' }, [photo, go]);
    var back = el('div', { class: 'face back', 'aria-hidden': 'true' }, [
      el('span', { class: 'tag', text: c.letter }),
      el('span', { html: ICON_X }),
      el('span', { class: 'back-txt', text: 'Ruled out' }),
      el('span', { class: 'back-why' })
    ]);
    return el('li', { class: 'card', 'data-c': c.id, 'aria-label': 'Card ' + c.letter }, [el('div', { class: 'card-inner' }, [front, back])]);
  }

  function renderStatus(lostHeart) {
    $('#q-count').textContent = String(G.asked.length);
    var h = $('#hearts');
    h.innerHTML = '';
    for (var i = 0; i < G.maxHearts; i++) {
      var empty = i >= G.hearts;
      var s = '<svg class="heart' + (empty ? ' empty' : '') + (lostHeart && i === G.hearts ? ' lost' : '') + '" viewBox="0 0 7 6" shape-rendering="crispEdges" aria-hidden="true">' + HEART + '</svg>';
      h.insertAdjacentHTML('beforeend', s);
    }
    h.setAttribute('aria-label', 'Patience: ' + G.hearts + ' of ' + G.maxHearts + ' hearts left');
    var p = $('#pips');
    p.innerHTML = '';
    G.cands.forEach(function (c) { p.appendChild(el('span', { class: 'pip' + (G.alive.indexOf(c.id) < 0 ? ' out' : '') })); });
    p.setAttribute('aria-label', G.alive.length + ' of ' + G.cands.length + ' cards left');
  }

  function flipOut(cid, why, delay) {
    var li = $('#cards [data-c="' + cid + '"]');
    if (!li) return;
    setTimeout(function () {
      li.classList.add('out');
      li.setAttribute('aria-label', 'Card ' + G.byId[cid].letter + ', ruled out');
      $('.back-why', li).textContent = why;
      var front = $('.front', li);
      front.setAttribute('inert', '');
      front.querySelectorAll('button').forEach(function (b) { b.disabled = true; });
      var m = el('span', { class: 'minus', text: '−1', 'aria-hidden': 'true' });
      li.appendChild(m);
      setTimeout(function () { m.remove(); }, 1100);
    }, delay);
  }

  function ask(qid) {
    if (!G || G.busy || G.over || G.hearts <= 0) return;
    var g = G;
    var q = g.qmap[qid];
    if (g.asked.some(function (a) { return a.qid === qid; })) return;
    var ans = q.truth[g.r.target];
    var removed = g.alive.filter(function (c) { return q.truth[c] !== ans; });
    g.alive = g.alive.filter(function (c) { return q.truth[c] === ans; });
    g.asked.push({ qid: qid, answer: ans, removed: removed, remaining: g.alive.slice() });
    if (g.alive.length === 1) stopClock(g); // solved: the rest is animation
    g.hearts -= 1;
    g.busy = true;
    var n = g.asked.length;

    var chip = $('#bank [data-q="' + qid + '"]');
    chip.disabled = true;
    chip.classList.add('asked');
    chip.setAttribute('aria-label', q.text + ' Answer: ' + (ans ? 'yes' : 'no') + (removed.length ? '' : '. Did not split anything.'));

    $('#msg-you-txt').textContent = q.text;
    $('#msg-you').hidden = false;
    $('#msg-human').hidden = true;
    var qc = $('#q-count');
    qc.classList.remove('bump'); void qc.offsetWidth; qc.classList.add('bump');

    wait(420).then(function () {
      if (g.dead) return;
      var mood = g.hearts <= 0 ? 'mad' : g.hearts <= Math.ceil(g.maxHearts / 3) ? 'meh' : 'happy';
      setMood(mood);
      $('#msg-human-txt').textContent = ans ? 'Yes.' : 'No.';
      $('#msg-human').hidden = false;
      chip.appendChild(el('span', { class: 'ans ' + (ans ? 'yes' : 'no'), text: ans ? 'Yes' : 'No' }));
      if (!removed.length) chip.appendChild(el('span', { class: 'dud', text: 'no split' }));
      logAsked(n, q.text, ans, removed.length);
      renderStatus(true);
      return wait(260);
    }).then(function () {
      if (g.dead) return;
      var why = 'Q' + n + ': ' + (ans ? 'Yes' : 'No');
      removed.forEach(function (c, i) { g.outBy[c] = n; flipOut(c, why, (reduceMQ.matches ? 0 : i * 110)); });
      if (removed.length) {
        toast('−' + plural(removed.length, 'card'), g.alive.length + ' left', 'good');
        announce('User says ' + (ans ? 'yes' : 'no') + '. ' + plural(removed.length, 'card') + ' ruled out. ' + g.alive.length + ' left. ' + plural(g.hearts, 'heart') + ' left.');
      } else {
        toast('That didn’t split anything', 'Every card left gives the same answer.', 'meh', 2800);
        announce('User says ' + (ans ? 'yes' : 'no') + '. That didn’t split anything: every card left gives the same answer. ' + plural(g.hearts, 'heart') + ' left.');
      }
      return wait(removed.length * 110 + 450);
    }).then(function () {
      if (g.dead) return;
      var left = pruneBank(g);
      return wait(left.pruned ? 340 : 0).then(function () { return left.count; });
    }).then(function (splitters) {
      if (g.dead) return;
      g.busy = false;
      if (g.alive.length === 1) {
        var last = g.alive[0];
        $('#cards [data-c="' + last + '"]').classList.add('win');
        lockBank();
        toast('Only one left!', 'Going to card ' + g.byId[last].letter + '…', 'good', 1600);
        return wait(1500).then(function () { if (!g.dead) finish(last, 'deduced'); });
      }
      if (g.hearts <= 0) {
        lockBank();
        setMood('mad');
        $('#msg-human-txt').textContent = (ans ? 'Yes.' : 'No.') + ' Enough questions. Just pick one!';
        $('#msg-human-txt').classList.add('long');
        toast('Out of patience!', 'Pick a card with Go here.', 'bad', 3000);
        announce('The user is out of patience. Pick a card.');
        var firstGo = $('#cards .card:not(.out) .go');
        if (firstGo) firstGo.focus();
        return;
      }
      if (!splitters) {
        // Only possible with a bank that cannot separate the cards left: let the player guess.
        lockBank();
        toast('No question can split these', 'Pick a card with Go here.', 'meh', 3000);
        announce('No question left can split the remaining cards. Pick a card.');
        var go1 = $('#cards .card:not(.out) .go');
        if (go1) go1.focus();
        return;
      }
      var first = $('#bank .chip:not([hidden]):not(:disabled)');
      if (first) first.focus();
    });
  }
  // Hide asked questions and every question on which all face-up cards agree.
  function pruneBank(g) {
    var askedIds = g.asked.map(function (a) { return a.qid; });
    var pruned = 0, count = 0;
    document.querySelectorAll('#bank .chip:not([hidden])').forEach(function (b) {
      var qid = b.getAttribute('data-q'), q = g.qmap[qid];
      var t0 = q.truth[g.alive[0]];
      var splits = g.alive.some(function (c) { return q.truth[c] !== t0; });
      if (askedIds.indexOf(qid) >= 0 || !splits) {
        pruned++;
        b.disabled = true;
        b.classList.add('leaving');
        setTimeout(function () { b.hidden = true; }, reduceMQ.matches ? 0 : 320);
      } else count++;
    });
    var h = $('#bank-h');
    h.textContent = count ? 'Ask the user' : 'No questions left';
    if (pruned && count) announce(plural(count, 'question') + ' can still split the cards.');
    return { pruned: pruned, count: count };
  }
  function logAsked(n, text, ans, removed) {
    var log = $('#asked-log');
    log.hidden = false;
    log.appendChild(el('li', {}, [
      el('span', { class: 'qn', text: 'Q' + n }),
      el('span', { class: 'lt', text: text }),
      el('span', { class: 'ans ' + (ans ? 'yes' : 'no'), text: ans ? 'Yes' : 'No' }),
      el('span', { class: 'rm' + (removed ? '' : ' dud'), text: removed ? '\u2212' + removed : 'no split' })
    ]));
  }
  function lockBank() {
    var bank = $('#bank');
    bank.classList.add('locked');
    bank.querySelectorAll('.chip').forEach(function (b) { b.disabled = true; });
  }

  // ---------- round clock: from cards shown to one card left / a pick; paused while the page is hidden ----------
  function startClock(g) { g.t0 = performance.now(); g.hiddenMs = 0; g.hiddenAt = document.hidden ? g.t0 : null; g.tSec = null; }
  function stopClock(g) {
    if (g.tSec != null || g.t0 == null) return;
    var now = performance.now();
    var hidden = g.hiddenMs + (g.hiddenAt != null ? now - g.hiddenAt : 0);
    g.tSec = Math.max(0, (now - g.t0 - hidden) / 1000);
  }
  document.addEventListener('visibilitychange', function () {
    var g = G;
    if (!g || g.t0 == null || g.tSec != null) return;
    var now = performance.now();
    if (document.hidden) { if (g.hiddenAt == null) g.hiddenAt = now; }
    else if (g.hiddenAt != null) { g.hiddenMs += now - g.hiddenAt; g.hiddenAt = null; }
  });

  function guess(cid) {
    if (!G || G.over || G.busy) return;
    if (G.alive.indexOf(cid) < 0) return;
    stopClock(G);
    closeLightbox();
    lockBank();
    G.busy = true;
    var g = G;
    var li = $('#cards [data-c="' + cid + '"]');
    li.classList.add(cid === g.r.target ? 'win' : 'shake');
    wait(cid === g.r.target ? 700 : 500).then(function () { if (!g.dead) finish(cid, 'guess'); });
  }

  // ---------- result ----------
  function finish(choice, how) {
    var g = G;
    g.over = true;
    var r = g.r, correct = choice === r.target;
    var you = g.asked.length, bot = g.robotQ;
    var outcome = !correct ? 'loss' : you < bot ? 'win' : you === bot ? 'tie' : 'loss';
    var lucky = correct && how === 'guess' && g.alive.length > 1;

    S.tally.played += 1; S.tally[outcome] += 1;
    var prev = S.done[g.id];
    var rank = { loss: 0, tie: 1, win: 2 };
    if (!prev || rank[outcome] > rank[prev]) S.done[g.id] = outcome;
    saveStore();

    var cat = r.category || 'object';
    var sY = 'You: ' + plural(you, 'question') + '.', sB = 'ProCompNav: ' + bot + '.';
    var head, line;
    if (!correct) {
      head = 'Wrong ' + cat + '!';
      line = 'You: wrong card. ' + sB + '\nProCompNav wins.';   // which cards: shown as pictures below
    } else if (outcome === 'win') {
      head = 'You beat the robot!';
      line = sY + ' ' + sB + '\nYou win!' + (lucky ? ' (Lucky guess: ' + g.alive.length + ' cards were still up.)' : '');
    } else if (outcome === 'tie') {
      head = 'Tie!';
      line = sY + ' ' + sB + '\nTie!' + (lucky ? ' (Lucky guess!)' : '');
    } else {
      head = 'ProCompNav wins.';
      line = sY + ' ' + sB + '\nProCompNav wins.';
    }
    var v = $('#verdict');
    v.className = 'verdict reveal-anim ' + (correct ? outcome : 'loss');
    $('#verdict-h').textContent = head;
    $('#verdict-line').textContent = line;
    $('#score-you').textContent = correct ? String(you) : '✕';
    $('#score-you').classList.toggle('bad', !correct);
    $('#score-you').setAttribute('aria-label', correct ? plural(you, 'question') : 'wrong pick');
    $('#score-bot').textContent = String(bot);

    // target photos
    var tp = $('#target-photos');
    tp.innerHTML = '';
    var T = g.byId[r.target];
    $('#target-cap').textContent = 'The user meant card ' + T.letter;
    if (r.target_photo && r.target_photo !== T.img) {
      tp.appendChild(el('figure', { class: 'tp' }, [el('img', { src: r.target_photo, alt: 'Photo of the target the user had in mind' }), el('figcaption', { text: 'What the user had in mind' })]));
    }
    tp.appendChild(el('figure', { class: 'tp' }, [el('img', { src: T.img, alt: 'Card ' + T.letter + ', the target' }), el('figcaption', { text: 'Card ' + T.letter + ' ✓ the target' })]));
    if (!correct) {
      var W = g.byId[choice];
      tp.appendChild(el('figure', { class: 'tp wrong' }, [el('img', { src: W.img, alt: 'Card ' + W.letter + ', your pick' }), el('figcaption', { text: 'Card ' + W.letter + ' ✕ your pick' })]));
    }

    buildPaths(choice, correct);
    var best = info(g.id).best_q;
    $('#best-note').textContent = (typeof best === 'number')
      ? 'Fewest questions possible with this bank: ' + best + '.' + (best < bot ? ' (The robot had to come up with its own questions; you got a ready-made list that only kept questions that still split the cards.)' : '')
      : '';
    if (g.asked.length && !g.asked[0].removed.length) {
      $('#best-note').textContent += ' Your first question split nothing. After that the game hid such questions; ProCompNav looks for a question that splits the pool every time.';
    }
    stopClock(g);
    renderTime(g);
    g.share = buildShare(g.id, correct ? outcome : 'wrong', you, bot);
    $('#share-box').hidden = true;
    $('#card-preview').hidden = true; $('#save-btn').hidden = true;
    makeResultCard(g, correct ? outcome : 'wrong', you, bot);
    renderTally($('#tally-result'), false);
    show('screen-result');
    $('#verdict-h').focus({ preventScroll: true });
    playReplay();
  }

  function renderTime(g) {
    // Time to pick questions. Not part of the score (that is the question count above): it only shows that
    // ProCompNav's RCJ picks a question that splits the cards in seconds, where people take a while.
    var box = $('#time-line'), pcn = g.r.procompnav || {};
    box.innerHTML = '';
    var you = g.tSec, bot = (typeof pcn.rcj_time_sec === 'number') ? pcn.rcj_time_sec : null;
    box.appendChild(el('div', { class: 'tr-h', text: 'Time to pick questions' }));
    function tile(cls, name, v) {
      return el('div', { class: 'tr-side ' + cls }, [
        el('span', { class: 'tr-name', text: name }),
        el('span', { class: 'tr-n', text: v.toFixed(1) + ' s' })
      ]);
    }
    var tiles = el('div', { class: 'tr-tiles' });
    tiles.appendChild(tile('you', 'You', you));
    if (bot != null) {
      tiles.appendChild(el('span', { class: 'tr-vs', 'aria-hidden': 'true', text: 'vs' }));
      tiles.appendChild(tile('bot', 'ProCompNav', bot));
    }
    box.appendChild(tiles);
    if (bot != null) {
      var ratio = you / Math.max(bot, 0.1);
      box.appendChild(el('p', { class: 'tr-cap', text: ratio >= 1.5
        ? 'ProCompNav chose its questions ' + Math.round(ratio) + '\u00d7 faster.'
        : (you < bot ? 'You picked faster this time.' : 'About as fast as ProCompNav.') }));
      var note = 'Not scored. Shows how fast RCJ picks questions' + (g.r.episode != null ? ' (measured, ep ' + g.r.episode + ').' : '.');
      box.appendChild(el('p', { class: 'tnote', text: note }));
    }
  }
  function tilesFor(aliveIds, final) {
    var g = G;
    var wrap = el('div', { class: 'tiles', role: 'img', 'aria-label': aliveIds.length + ' left: ' + aliveIds.map(function (c) { return g.byId[c].letter; }).join(', ') });
    g.cands.forEach(function (c) {
      var isOut = aliveIds.indexOf(c.id) < 0;
      wrap.appendChild(el('div', { class: 'tile' + (isOut ? ' out' : '') + (final && c.id === g.r.target ? ' target' : ''), 'aria-hidden': 'true' }, [
        el('img', { src: c.img, alt: '', loading: 'lazy' }), el('span', { text: c.letter })
      ]));
    });
    return wrap;
  }
  function stepEl(n, text, ans, aliveIds, prevCount, final) {
    var removed = prevCount - aliveIds.length;
    return el('li', { class: 'step' }, [
      el('div', { class: 'step-q' }, [
        el('span', { class: 'qn', text: 'Q' + n }),
        el('span', { text: text }),
        el('span', { class: 'ans ' + (ans ? 'yes' : 'no'), text: ans ? 'Yes' : 'No' })
      ]),
      el('div', { class: 'step-meta' }, [
        tilesFor(aliveIds, final),
        el('span', { class: 'step-left' + (removed ? '' : ' dud'), text: removed ? '−' + removed + ' → ' + aliveIds.length + ' left' : 'no split' })
      ])
    ]);
  }
  function startEl() {
    var all = G.cands.map(function (c) { return c.id; });
    return el('li', { class: 'step' }, [
      el('div', { class: 'step-q' }, [el('span', { text: 'Start: ' + plural(all.length, 'card') })]),
      el('div', { class: 'step-meta' }, [tilesFor(all, false)])
    ]);
  }
  function endEl(choice, ok) {
    var c = G.byId[choice];
    return el('li', { class: 'step final' + (ok ? '' : ' bad') }, [
      el('div', { class: 'step-q' }, [el('span', { text: 'Went to card ' + c.letter + (ok ? ' ✓' : ' ✕ wrong') })])
    ]);
  }
  function buildPaths(choice, correct) {
    var g = G, ys = $('#steps-you'), bs = $('#steps-bot');
    ys.innerHTML = ''; bs.innerHTML = '';
    ys.appendChild(startEl()); bs.appendChild(startEl());
    var prev = g.cands.length;
    g.asked.forEach(function (a, i) {
      ys.appendChild(stepEl(i + 1, g.qmap[a.qid].text, a.answer, a.remaining, prev, i === g.asked.length - 1 && a.remaining.length === 1));
      prev = a.remaining.length;
    });
    ys.appendChild(endEl(choice, correct));
    prev = g.cands.length;
    g.r.procompnav.steps.forEach(function (s, i, arr) {
      var q = g.qmap[s.question_id];
      bs.appendChild(stepEl(i + 1, q ? q.text : s.attr, s.answer, s.remaining, prev, i === arr.length - 1));
      prev = s.remaining.length;
    });
    bs.appendChild(endEl(g.r.target, true));
  }
  var replayToken = 0;
  function playReplay() {
    var tok = ++replayToken;
    var ys = $('#steps-you').children, bs = $('#steps-bot').children;
    var n = Math.max(ys.length, bs.length);
    for (var i = 0; i < n; i++) { if (ys[i]) ys[i].classList.add('pending'); if (bs[i]) bs[i].classList.add('pending'); }
    var step = function (i) {
      if (tok !== replayToken || i >= n) return;
      if (ys[i]) ys[i].classList.remove('pending');
      if (bs[i]) bs[i].classList.remove('pending');
      setTimeout(function () { step(i + 1); }, reduceMQ.matches ? 0 : 650);
    };
    setTimeout(function () { step(0); }, reduceMQ.matches ? 0 : 450);
  }
  $('#replay-btn').addEventListener('click', playReplay);
  $('#again-btn').addEventListener('click', function () { if (G) startRound(G.id); });
  $('#next-btn').addEventListener('click', function () {
    var cur = G ? G.id : null, cat = G ? G.r.category : null;
    var real = realRounds();
    var inCat = (manifest.categories && manifest.categories[cat]) || [];
    var unplayedCat = inCat.filter(function (id) { return !S.done[id] && id !== cur; });
    var next = unplayedCat.length ? pickFrom(unplayedCat) : real.length ? pickFrom(real, cur) : cur;
    if (next) startRound(next);
  });

  // ---------- result card (canvas, 1200x630) ----------
  var PIX = {
    trophy: ['................', '..KKKKKKKKKKKK..', 'KKKYYHYYYYYYDKKK', 'KYKYYHYYYYYYDKYK', 'KYKYYHYYYYYYDKYK', 'KYKYYHYYYYYYDKYK',
      '.KKYYHYYYYYYDKK.', '..KKYYYYYYYDKK..', '...KKYYYYYDKK...', '.....KKYYKK.....', '......KYDK......', '......KYDK......',
      '.....KKYDKK.....', '....KYYYYYDK....', '...KKKKKKKKKK...', '................'],
    shake: ['................', '................', '................', 'GGG..........OOO', 'GGGG........OOOO', 'GGGGKSSSSSSKOOOO',
      'GGGKSSSSSSSSKOOO', 'GGKSSKSSKSSSSKOO', 'GKSSSSKSSKSSSSKO', '..KSSSSKSSKSSK..', '...KSSSSSSSSK...', '....KKSSSSKK....',
      '......KKKK......', '................', '................', '................']
  };
  var PIXC = { K: '#1b241d', Y: '#f0ad1f', H: '#ffe08a', D: '#b37a06', G: '#46a035', O: '#f0ad1f', S: '#e9b98c' };
  function drawPix(ctx, name, x, y, s) {
    if (name === 'robot') {
      robotRects('#f0ad1f', '#9a6a00', '#fff2b8').forEach(function (r) { ctx.fillStyle = r[4]; ctx.fillRect(x + r[0] * s, y + r[1] * s, r[2] * s, r[3] * s); });
      return;
    }
    PIX[name].forEach(function (row, j) {
      for (var i = 0; i < row.length; i++) { var c = PIXC[row[i]]; if (c) { ctx.fillStyle = c; ctx.fillRect(x + i * s, y + j * s, s, s); } }
    });
  }
  function rrect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function blockBox(ctx, x, y, w, h, fill, r, sh, bw, edge) {
    ctx.fillStyle = edge || '#1b241d'; rrect(ctx, x, y + sh, w, h, r); ctx.fill();
    ctx.fillStyle = fill; rrect(ctx, x, y, w, h, r); ctx.fill();
    ctx.lineWidth = bw; ctx.strokeStyle = edge || '#1b241d'; rrect(ctx, x + bw / 2, y + bw / 2, w - bw, h - bw, Math.max(1, r - bw / 2)); ctx.stroke();
  }
  function fitText(ctx, text, maxW, size, fam) {
    var s = size;
    do { ctx.font = fam.replace('%', s); s -= 2; } while (ctx.measureText(text).width > maxW && s > 12);
  }
  function makeResultCard(g, res, you, bot) {
    var fontsReady = (document.fonts && document.fonts.load)
      ? Promise.all([document.fonts.load('96px "Jersey 10"'), document.fonts.load('900 40px Nunito'), document.fonts.load('800 24px Nunito')]).catch(function () {})
      : Promise.resolve();
    var timeout = new Promise(function (r) { setTimeout(r, 2500); });
    Promise.race([fontsReady, timeout]).then(function () {
      if (g.dead || G !== g) return;
      var cv;
      try { cv = drawResultCard(res, you, bot); } catch (e) { return; }
      g.cardCanvas = cv;
      try {
        cv.toBlob(function (blob) {
          if (!blob || G !== g) return;
          g.cardFile = (typeof File === 'function') ? new File([blob], 'you-vs-procompnav.png', { type: 'image/png' }) : null;
          g.cardURL = URL.createObjectURL(blob);
          var pv = $('#card-preview');
          pv.src = g.cardURL;
          pv.hidden = false;
          $('#save-btn').hidden = false;
        }, 'image/png');
      } catch (e) { /* tainted canvas or no toBlob: text-only share */ g.cardFile = null; }
    });
  }
  // Scoreboard card. Same layout as tools/make_share_previews.py (keep them in sync).
  function drawResultCard(res, you, bot) {
    var W = 1200, H = 630, INK = '#1b241d';
    var cv = document.createElement('canvas'); cv.width = W; cv.height = H;
    var ctx = cv.getContext('2d');
    var PX = '"Jersey 10", "Nunito", system-ui, sans-serif', UI = 'Nunito, system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.fillStyle = '#eaf1e4'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#dbe6d3';
    for (var yy = 8; yy < H; yy += 24) for (var xx = 8; xx < W; xx += 24) ctx.fillRect(xx, yy, 3, 3);

    // top: badge + result line
    var badgeBg = res === 'win' ? '#e0f3d8' : res === 'tie' ? '#ffffff' : '#fbe1de';
    blockBox(ctx, 60, 24, 152, 152, badgeBg, 16, 8, 6);
    drawPix(ctx, res === 'win' ? 'trophy' : res === 'tie' ? 'shake' : 'robot', 72, 36, 8);
    var head = res === 'win' ? 'I beat the robot!' : res === 'tie' ? 'Tied with the robot.' : 'The robot won.';
    ctx.fillStyle = res === 'win' ? '#2c6e20' : res === 'tie' ? INK : '#b3261e';
    fitText(ctx, head, 900, 128, '%px ' + PX);
    ctx.fillText(head, 244, res === 'wrong' ? 118 : 138);
    if (res === 'wrong') {
      ctx.font = '900 30px ' + UI;
      var wt = 'Picked the wrong one', ww = ctx.measureText(wt).width + 28;
      ctx.fillStyle = '#d63f35'; rrect(ctx, 246, 132, ww, 44, 8); ctx.fill();
      ctx.fillStyle = '#ffffff'; ctx.fillText(wt, 260, 164);
    }

    // middle: heading + two-row scoreboard
    ctx.fillStyle = '#536257'; ctx.font = '900 30px ' + UI;
    ctx.fillText('Questions asked · fewer wins', 64, 214);
    var youWins = res === 'win', botWins = res === 'loss' || res === 'wrong', tie = res === 'tie';
    var rows = [
      { label: 'YOU', who: 'you', n: you, hi: youWins || tie, fill: '#e0f3d8', edge: '#2c6e20', blk: '#46a035', blkHi: '#8fd873', blkLo: '#2c6e20', wrong: res === 'wrong' },
      { label: 'ROBOT', who: 'robot', n: bot, hi: botWins || tie, fill: '#fdefc7', edge: '#9a6a00', blk: '#f0ad1f', blkHi: '#ffe08a', blkLo: '#9a6a00', wrong: false }
    ];
    rows.forEach(function (r, i) {
      var x = 60, y = 228 + i * 130, w = 1080, h = 114;
      blockBox(ctx, x, y, w, h, r.hi ? r.fill : '#f4f7f0', 14, 7, r.hi ? 7 : 4, r.hi ? r.edge : INK);
      // avatar + label
      if (r.who === 'you') robotRects('#46a035', '#2c6e20', '#b8f59a').forEach(function (q) { ctx.fillStyle = q[4]; ctx.fillRect(x + 22 + q[0] * 5, y + 17 + q[1] * 5, q[2] * 5, q[3] * 5); });
      else drawPix(ctx, 'robot', x + 22, y + 17, 5);
      ctx.fillStyle = INK; ctx.font = '72px ' + PX; ctx.textAlign = 'left';
      ctx.fillText(r.label, x + 116, y + 80);
      // big number + unit
      var nx = x + 290;
      blockBox(ctx, nx, y + 12, 104, 90, r.wrong ? '#d63f35' : r.blk, 10, 5, 4);
      ctx.fillStyle = r.who === 'you' || r.wrong ? '#ffffff' : INK; ctx.font = '96px ' + PX; ctx.textAlign = 'center';
      ctx.fillText(String(r.n), nx + 52, y + 90);
      ctx.textAlign = 'left';
      var unit = r.n === 1 ? 'question' : 'questions';
      ctx.fillStyle = INK; ctx.font = '900 42px ' + UI;
      ctx.fillText(unit, nx + 120, y + 72);
      // one chunky block per question (shrinks to fit)
      var bx0 = Math.max(nx + 120 + ctx.measureText('questions').width + 30, x + 640), room = x + w - 24 - bx0;
      var bs = Math.min(60, Math.floor((room + 12) / Math.max(1, r.n) - 12)), gap = Math.max(6, Math.round(bs * 0.2));
      var t = Math.max(4, Math.round(bs * 0.16));
      for (var k = 0; k < r.n; k++) {
        var bx = bx0 + k * (bs + gap), by = y + (h - bs) / 2;
        ctx.fillStyle = INK; ctx.fillRect(bx, by + 5, bs, bs);
        ctx.fillStyle = r.blk; ctx.fillRect(bx, by, bs, bs);
        ctx.fillStyle = r.blkHi; ctx.fillRect(bx, by, bs, t); ctx.fillRect(bx, by, t, bs);
        ctx.fillStyle = r.blkLo; ctx.fillRect(bx, by + bs - t, bs, t);
        ctx.strokeStyle = INK; ctx.lineWidth = 4; ctx.strokeRect(bx + 2, by + 2, bs - 4, bs - 4);
      }
      if (r.wrong) {
        // sticker on the row's top-right corner, so it never takes room from the blocks
        ctx.font = '900 26px ' + UI;
        var tw = ctx.measureText('WRONG PICK').width + 28;
        ctx.fillStyle = INK; rrect(ctx, x + w - 24 - tw, y - 18 + 4, tw, 38, 8); ctx.fill();
        ctx.fillStyle = '#d63f35'; rrect(ctx, x + w - 24 - tw, y - 18, tw, 38, 8); ctx.fill();
        ctx.fillStyle = '#ffffff'; ctx.fillText('WRONG PICK', x + w - 24 - tw + 14, y + 10);
      }
    });

    // bottom
    ctx.textAlign = 'center'; ctx.fillStyle = INK;
    ctx.font = '900 40px ' + UI;
    ctx.fillText('Can you beat a NeurIPS 2026 robot?', W / 2, 550);
    ctx.fillStyle = INK; ctx.fillRect(0, H - 44, W, 44);
    ctx.fillStyle = '#e8f0e9'; ctx.font = '800 24px ' + UI;
    ctx.fillText('You vs ProCompNav · NeurIPS 2026 Oral', W / 2, H - 14);
    ctx.textAlign = 'left';
    ctx.fillStyle = '#8a5a35'; ctx.fillRect(36, H - 37, 30, 30);
    ctx.fillStyle = '#6cc04f'; ctx.fillRect(39, H - 34, 24, 8);
    return cv;
  }

  // ---------- share ----------
  var shareIndex = null; // s/index.json: { rounds: { id: { max_y: N } } }
  function shareBase() {
    var b = (typeof SHARE_BASE_URL === 'string' && SHARE_BASE_URL) ? SHARE_BASE_URL : location.href;
    b = b.split('#')[0];
    if (SHARE_BASE_URL && !/\/$/.test(b) && !/\.html?$/.test(b)) b += '/';
    return b;
  }
  function shareLink(id, res, you) {
    var base = shareBase();
    var s = shareIndex && shareIndex.rounds && shareIndex.rounds[id];
    try {
      if (s && you <= s.max_y) {
        return new URL('s/' + id + (res === 'wrong' ? '-wrong-y' : '-y') + you + '.html', base).href;
      }
    } catch (e) { /* bad base: fall through */ }
    return base + '#r=' + encodeURIComponent(id);
  }
  function buildShare(id, res, you, bot) {
    var counts = ' (You ' + you + ' \u00b7 Robot ' + bot + ' questions)';
    var line = res === 'win' ? '\ud83c\udfc6 I beat the robot!' + counts
      : res === 'tie' ? '\ud83e\udd1d Tied with the robot.' + counts
      : res === 'wrong' ? '\ud83e\udd16 The robot won. \ud83e\udd72 (You picked the wrong one)'
      : '\ud83e\udd16 The robot won. \ud83e\udd72' + counts;
    var url = shareLink(id, res, you);
    var body = 'You vs ProCompNav\n\n' + line + '\n\nCan you beat a NeurIPS 2026 robot?';
    return { title: 'You vs ProCompNav', body: body, url: url, full: body + '\n' + url };
  }
  function showShareBox(text) {
    var box = $('#share-box'), ta = $('#share-text');
    ta.value = text;
    box.hidden = false;
    ta.focus();
    ta.select();
  }
  function copyShare(text) {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
          toast('Copied!', 'Paste it to a friend.', 'good');
          announce('Result copied to the clipboard.');
        }, function () { showShareBox(text); });
        return;
      }
    } catch (e) { /* fall through */ }
    showShareBox(text);
  }
  $('#save-btn').addEventListener('click', function () {
    var g = G;
    if (!g || !g.cardURL) return;
    try {
      var a = el('a', { href: g.cardURL, download: 'you-vs-procompnav-' + g.id + '.png' });
      document.body.appendChild(a); a.click(); a.remove();
      toast('Saved', 'Check your downloads.', 'good', 1600);
    } catch (e) { toast('Could not save', 'Long-press the picture instead.', 'meh'); }
  });
  $('#share-btn').addEventListener('click', function () {
    var d = G && G.share;
    if (!d) return;
    if (navigator.share) {
      // The link goes in `url`; most share targets append it after the text.
      var data = { title: d.title, text: d.body, url: d.url };
      var file = G.cardFile;
      try {
        if (file && navigator.canShare && navigator.canShare({ files: [file] })) data.files = [file];
      } catch (e) { /* canShare can throw on odd input */ }
      var p;
      try { p = navigator.share(data); } catch (e) { p = Promise.reject(e); }
      Promise.resolve(p).catch(function (e) {
        if (e && e.name === 'AbortError') return; // user closed the share sheet
        copyShare(d.full);
      });
      return;
    }
    copyShare(d.full);
  });

  // ---------- lightbox ----------
  var LB = { cid: null, i: 0 };
  var dlg = $('#lightbox');
  function openLightbox(cid, i) {
    if (!G) return;
    LB.cid = cid; LB.i = i || 0;
    renderLightbox();
    if (typeof dlg.showModal === 'function') { if (!dlg.open) dlg.showModal(); } else dlg.setAttribute('open', '');
  }
  function renderLightbox() {
    var c = G.byId[LB.cid], nv = c.views.length;
    $('#lb-title').textContent = 'Card ' + c.letter;
    var img = $('#lb-img');
    img.src = c.views[LB.i];
    img.alt = 'Card ' + c.letter + ', view ' + (LB.i + 1) + ' of ' + nv;
    $('#lb-count').textContent = nv > 1 ? 'View ' + (LB.i + 1) + ' of ' + nv : 'Only one view';
    $('#lb-prev').disabled = nv < 2; $('#lb-next').disabled = nv < 2;
    $('#lb-prev').hidden = nv < 2; $('#lb-next').hidden = nv < 2;
    var th = $('#lb-thumbs');
    th.innerHTML = '';
    th.hidden = nv < 2;
    if (nv > 1) c.views.forEach(function (v, j) {
      var b = el('button', { type: 'button', 'aria-label': 'View ' + (j + 1), 'aria-current': j === LB.i ? 'true' : 'false' }, [el('img', { src: v, alt: '' })]);
      b.addEventListener('click', function () { LB.i = j; renderLightbox(); });
      th.appendChild(b);
    });
    var canGo = !G.over && !G.busy && G.alive.indexOf(LB.cid) >= 0;
    $('#lb-go').hidden = !canGo;
    $('#lb-go').textContent = 'Go to card ' + c.letter;
  }
  function closeLightbox() { if (dlg.open) { if (typeof dlg.close === 'function') dlg.close(); else dlg.removeAttribute('open'); } }
  function stepLB(d) { var nv = G.byId[LB.cid].views.length; LB.i = (LB.i + d + nv) % nv; renderLightbox(); }
  $('#lb-prev').addEventListener('click', function () { stepLB(-1); });
  $('#lb-next').addEventListener('click', function () { stepLB(1); });
  $('#lb-close').addEventListener('click', closeLightbox);
  $('#lb-close2').addEventListener('click', closeLightbox);
  $('#lb-go').addEventListener('click', function () { var c = LB.cid; closeLightbox(); guess(c); });
  dlg.addEventListener('click', function (e) { if (e.target === dlg) closeLightbox(); });
  dlg.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowLeft') { e.preventDefault(); stepLB(-1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); stepLB(1); }
  });
  dlg.addEventListener('close', function () {
    var li = LB.cid && $('#cards [data-c="' + LB.cid + '"] .photo');
    if (li && !li.disabled && !$('#screen-round').hidden) li.focus();
  });

  // ---------- boot ----------
  function showLoadError(msg) {
    var m = $('#load-msg');
    m.textContent = msg;
    m.hidden = false;
    show('screen-title');
  }
  paintAvatars();
  fetchJSON('s/index.json').then(function (si) { shareIndex = si; }).catch(function () { /* no preview pages: share #r= links */ });
  fetchJSON('data/manifest.json').then(function (m) {
    manifest = m;
    manifest.rounds = manifest.rounds || {};
    renderTitle();
    var h = (location.hash || '').match(/^#r=([^&]+)/);
    var want = h ? decodeURIComponent(h[1]) : null;
    if (want && manifest.rounds[want]) startRound(want);
    else show('screen-title');
    window.addEventListener('hashchange', function () {
      var m2 = (location.hash || '').match(/^#r=([^&]+)/);
      var id2 = m2 ? decodeURIComponent(m2[1]) : null;
      if (id2 && manifest.rounds[id2] && !(G && G.id === id2 && !G.over)) startRound(id2);
    });
    if (!Object.keys(manifest.rounds).length) showLoadError('No rounds yet. Run tools/build_manifest.py to list them.');
  }).catch(function () {
    showLoadError(location.protocol === 'file:'
      ? 'Browsers block loading game data from a local file. Serve this folder instead: run  python3 -m http.server  inside it and open http://localhost:8000/.'
      : 'Could not load data/manifest.json.');
  });
})();
