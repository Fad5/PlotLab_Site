const datasets = {{ datasets|default:"[]"|safe }};
const body = document.body;
const themeToggle = document.getElementById("themeToggle");
function getPlotlyLayoutOverrides(theme) {
  const styles = getComputedStyle(document.body);
  return {
    template: theme === "dark" ? "plotly_dark" : "plotly_white",
    font: { color: styles.getPropertyValue("--text-color").trim() },
    plot_bgcolor: styles.getPropertyValue("--box-bg").trim(),
    paper_bgcolor: styles.getPropertyValue("--box-bg").trim(),
    xaxis: {
      gridcolor: styles.getPropertyValue("--grid-color").trim(),
      color: styles.getPropertyValue("--text-color").trim()
    },
    yaxis: {
      gridcolor: styles.getPropertyValue("--grid-color").trim(),
      color: styles.getPropertyValue("--text-color").trim()
    }
  };
}
function drawPlots() {
  const theme = body.classList.contains("dark") ? "dark" : "light";
  const container = document.getElementById("plot-container");
  container.innerHTML = "";  // Очистка
  if (!Array.isArray(datasets) || datasets.length === 0) return;
  datasets.forEach((plot, index) => {
    const div = document.createElement("div");
    div.className = "plot";
    div.id = `plot-${index}`;
    container.appendChild(div);
    Plotly.newPlot(div.id, [{
      x: plot.x,
      y: plot.y,
      type: 'scatter',
      mode: 'lines',
      marker: { color: getComputedStyle(document.body).getPropertyValue("--plot-color").trim() }
    }], {
      title: plot.title,
      xaxis: { title: plot.xaxis_title },
      yaxis: { title: plot.yaxis_title },
      margin: { t: 50, l: 50, r: 50, b: 50 },
      ...getPlotlyLayoutOverrides(theme)
    }, { responsive: true });
  });
}
function applyTheme(theme) {
  if (theme === "dark") {
    body.classList.add("dark");
    themeToggle.textContent = "☀️ Светлая тема";
  } else {
    body.classList.remove("dark");
    themeToggle.textContent = "🌙 Тёмная тема";
  }
  drawPlots();
}
const savedTheme = localStorage.getItem("theme") || "dark";
applyTheme(savedTheme);
themeToggle.addEventListener("click", () => {
  const newTheme = body.classList.contains("dark") ? "light" : "dark";
  localStorage.setItem("theme", newTheme);
  applyTheme(newTheme);
});
document.addEventListener("DOMContentLoaded", drawPlots);
