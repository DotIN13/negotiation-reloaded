// ─── Constants & State ──────────────────────────────────────────────────────
const socket = io();
const CHART_IDS = {
  conflict: "conflict-chart",
  issuesLine: "issues-chart",
  issuesStacked: "issue-stacked-chart",
};
let issueNames = [];
let issueLineTraces = [];
let battleMarkers = {};

// pick six distinct colors (Plotly’s Category10, for example)
const issueColorPairs = [
  { A: '#1f78b4', B: '#a6cee3' }, // issue 1
  { A: '#33a02c', B: '#b2df8a' }, // issue 2
  { A: '#e31a1c', B: '#fb9a99' }, // issue 3
  { A: '#ff7f00', B: '#fdbf6f' }, // issue 4
  { A: '#6a3d9a', B: '#cab2d6' }, // issue 5
  { A: '#b15928', B: '#fbb4a8' }, // issue 6
];

const layouts = {
  conflict: {
    title: { text: 'Conflict Intensity Over Time', font: { size: 16 } },
    xaxis: { title: "Step" },
    yaxis: { title: "Conflict Intensity" },
    margin: { t: 50, r: 20, b: 50, l: 50 },
  },
  issuesLine: {
    title: { text: 'Issue Shares Over Time (Camp A)', font: { size: 16 } },
    xaxis: { title: "Step" },
    yaxis: { title: "Share %" },
    margin: { t: 50, r: 20, b: 20, l: 50 },
  },
  issuesStacked: {
    title: { text: 'Issue Shares (Camp A vs. Camp B)', font: { size: 16 } },
    barmode: "stack",
    orientation: "h",
    margin: { t: 50, r: 20, b: 40, l: 50 },
    xaxis: { title: "Share" },
    yaxis: { title: "Issue", automargin: true, autorange: "reversed", tickpadding: 20 },
  },
};

// ─── Utility Functions ──────────────────────────────────────────────────────
function to_title(str) {
  return str.
    split('_').
    map(w => w[0].toUpperCase() + w.substr(1).toLowerCase()).
    join(' ');
}

// ─── Initialization ─────────────────────────────────────────────────────────
function initCharts() {
  Plotly.newPlot(
    CHART_IDS.conflict,
    [
      {
        x: [],
        y: [],
        mode: "lines",
        name: "Conflict",
      },
    ],
    layouts.conflict
  );

  Plotly.newPlot(CHART_IDS.issuesLine, [], layouts.issuesLine);

  Plotly.newPlot(CHART_IDS.issuesStacked, [], layouts.issuesStacked);
}

function initMap() {
  const map = L.map("map");
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  // compute bounds
  const battlefieldCoords = [
    [43.8564, 18.4130],  // Sarajevo
    [43.3436, 17.8075],  // Mostar
    [44.1042, 19.2972],  // Srebrenica
    [44.5381, 18.6761],  // Tuzla
    [44.8147, 15.8692],  // Bihac
    [43.6686, 18.9756],  // Gorazde
    [44.7725, 17.1925],  // Banja Luka
    [44.8772, 18.8111]   // Brcko
  ];

  const bounds = L.latLngBounds(battlefieldCoords);
  map.fitBounds(bounds, { padding: [20, 20] });  // optional padding for a nice margin

  return map;
}

const map = initMap();

// ─── Reset before re-configuring ─────────────────────────────────────────────
function resetVisuals() {
  issueNames = [];
  issueLineTraces = [];
  Object.values(battleMarkers).forEach((m) => m.remove());
  battleMarkers = {};

  // reset charts
  Plotly.react(
    CHART_IDS.conflict,
    [
      {
        x: [],
        y: [],
        mode: "lines",
        name: "Conflict",
      },
    ],
    layouts.conflict
  );

  Plotly.react(CHART_IDS.issuesLine, [], layouts.issuesLine);
  Plotly.react(CHART_IDS.issuesStacked, [], layouts.issuesStacked);
}

// ─── Handlers ────────────────────────────────────────────────────────────────
function onConfigSubmit(event) {
  event.preventDefault();

  // Disable the button to prevent multiple submissions
  const btn = document.getElementById('configure-btn');
  btn.disabled = true;
  btn.textContent = 'Running…';

  // 1) unbind the old step_update listener
  socket.off('step_update', onStepUpdate);

  // 2) clear everything
  resetVisuals();

  // 3) send new config
  const data = [...new FormData(event.target)].reduce(
    (acc, [k,v]) => ({ ...acc, [k]: +v }), {}
  );
  socket.emit('start_simulation', data);

  // 4) bind fresh listener for only this run
  socket.on('step_update', onStepUpdate);
}

function updateConflict({ step, conflict_intensity }) {
  Plotly.extendTraces(
    CHART_IDS.conflict,
    { x: [[step]], y: [[conflict_intensity]] },
    [0]
  );
}

function updateIssuesLine({ step, proposals }) {
  if (issueNames.length === 0) {
    proposals.forEach(({ issue }, i) => {
      issueNames.push(issue);
      issueLineTraces.push({
        x: [],
        y: [],
        mode: 'lines',
        name: `${to_title(issue)} (A)`,
        line: { color: issueColorPairs[i].A, width: 2 },
        opacity: 0.8
      });
    });
    Plotly.react(CHART_IDS.issuesLine, issueLineTraces, layouts.issuesLine);
  }

  proposals.forEach(({ A = 0 }, i) => {
    Plotly.extendTraces(
      CHART_IDS.issuesLine,
      { x: [[step]], y: [[A]] },
      [i]
    );
  });
}

function updateIssuesStacked({ proposals }) {
  const issues      = proposals.map(p => to_title(p.issue));
  const aShares     = proposals.map(p => p.A || 0);
  const bShares     = proposals.map(p => p.B || 0);
  const totalShares = aShares.map((a,i) => a + bShares[i]);
  const acceptedText = proposals.map(p =>
    Array.isArray(p.accepted_by) ? p.accepted_by.join(', ') : ''
  );

  // extract exactly as many colors as there are issues
  const colorsA = issueColorPairs.slice(0, issues.length).map(pair => pair.A);
  const colorsB = issueColorPairs.slice(0, issues.length).map(pair => pair.B);

  const traceA = {
    x: aShares,
    y: issues,
    name: 'A',
    type: 'bar',
    orientation: 'h',
    showlegend: false,
    marker: {
      color: colorsA,
      line: { width: 1 }
    }
  };

  const traceB = {
    x: bShares,
    y: issues,
    name: 'B',
    type: 'bar',
    orientation: 'h',
    showlegend: false,
    marker: {
      color: colorsB,
      line: { width: 1 }
    }
  };

  const traceText = {
    x: totalShares.map(t => t/2),
    y: issues,
    text: acceptedText,
    mode: 'text',
    orientation: 'h',
    textfont: { color: 'white', size: 12 },
    showlegend: false,
    hoverinfo: 'none'
  };

  Plotly.react(
    CHART_IDS.issuesStacked,
    [ traceA, traceB, traceText ],
    layouts.issuesStacked
  );
}

function updateMap({ battles }) {
  battles.forEach(({ name, lat, lng, intensity }) => {
    const radius = 10_000 * Math.sqrt(intensity);

    if (!battleMarkers[name]) {
      battleMarkers[name] = L.circle([lat, lng], {
        radius,
        fillOpacity: 0.5
      })
      .addTo(map)
      .bindTooltip(name, { permanent: true, direction: 'center', className: 'battle-label' });
    } else {
      battleMarkers[name].setRadius(radius);
    }
  });
}

function onStepUpdate(data) {
  updateConflict(data);
  updateIssuesLine(data);
  updateIssuesStacked(data);
  updateMap(data);
}

// ─── Socket.IO wiring ───────────────────────────────────────────────────────
socket.on("step_update", onStepUpdate);
socket.on("simulation_complete", ({ total_steps }) => {
  alert(`Simulation finished in ${total_steps} steps.`);
  const btn = document.getElementById('configure-btn');
  btn.disabled = false;
  btn.textContent = 'Configure & Start';
});

// ─── Kickoff ────────────────────────────────────────────────────────────────
document
  .getElementById("config-form")
  .addEventListener("submit", onConfigSubmit);

initCharts();
