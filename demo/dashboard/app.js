// EdgeSpark demo client. Opens two SSE streams, vanilla and EdgeSpark, races
// them side by side, and draws the per-round draft-depth / accepted-length strip.

const $ = (id) => document.getElementById(id);
const BLOCK = 5;

let sources = [];

function reset() {
  sources.forEach((s) => s.close());
  sources = [];
  $("out-vanilla").innerHTML = "";
  $("out-edge").innerHTML = "";
  $("depth").innerHTML = "";
  $("s-vanilla").innerHTML = ", <small>tok/s</small>";
  $("s-edge").innerHTML = ", <small>tok/s</small>";
  $("s-speedup").textContent = ", ";
}

function appendTokens(el, tokens, kind) {
  tokens.forEach((t, i) => {
    const span = document.createElement("span");
    // The last token of an EdgeSpark round is the verifier's token (bonus/correction).
    const isBonus = kind === "edge" && i === tokens.length - 1;
    span.className = "tok " + (isBonus ? "bonus-tok" : "accepted-tok");
    span.textContent = t + " ";
    el.appendChild(span);
  });
  el.scrollTop = el.scrollHeight;
}

function drawRound(ell, accepted) {
  const round = document.createElement("div");
  round.className = "round";
  for (let j = 0; j < BLOCK; j++) {
    const cell = document.createElement("div");
    if (j >= ell) cell.className = "cell skipped";
    else if (j < accepted) cell.className = "cell accepted";
    else cell.className = "cell bonus";
    round.appendChild(cell);
  }
  $("depth").appendChild(round);
}

const speed = { vanilla: 0, edge: 0 };
function updateSpeedup() {
  if (speed.vanilla > 0 && speed.edge > 0) {
    const r = speed.edge / speed.vanilla;
    $("s-speedup").textContent = "+" + Math.round((r - 1) * 100) + "%";
  }
}

function stream(variant, n, seed, outEl, scoreEl, key) {
  return new Promise((resolve) => {
    const src = new EventSource(`/api/stream?variant=${variant}&n=${n}&seed=${seed}`);
    sources.push(src);
    src.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      appendTokens(outEl, ev.tokens, key);
      speed[key] = ev.tok_s;
      scoreEl.innerHTML = `${ev.tok_s.toFixed(1)} <small>tok/s</small>`;
      if (key === "edge") drawRound(ev.ell, ev.accepted);
      updateSpeedup();
    };
    src.addEventListener("done", () => { src.close(); resolve(); });
    src.onerror = () => { src.close(); resolve(); };
  });
}

$("run").addEventListener("click", async () => {
  reset();
  const n = parseInt($("ntokens").value, 10) || 96;
  const seed = parseInt($("seed").value, 10) || 0;
  $("run").disabled = true;
  await Promise.all([
    stream("vanilla", n, seed, $("out-vanilla"), $("s-vanilla"), "vanilla"),
    stream("edgespark", n, seed, $("out-edge"), $("s-edge"), "edge"),
  ]);
  $("run").disabled = false;
});
