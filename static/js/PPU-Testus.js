document.addEventListener('DOMContentLoaded', function() {
    // Получаем элементы темы
    const body = document.body;
    const themeToggle = document.getElementById("themeToggle");
    
    // Парсим данные из Django
    const datasets = JSON.parse('{{ datasets_json|escapejs }}' || '[]');
    const mainPlots = JSON.parse('{{ main_plots_json|escapejs }}' || '{}');
    const summaryData = JSON.parse('{{ summary_data_json|escapejs }}' || '{}');
    
    console.log('Loaded data:', {datasets, mainPlots, summaryData});

    // Функция для получения настроек темы Plotly
    function getPlotlyLayoutOverrides(theme) {
        const styles = getComputedStyle(document.body);
        return {
            template: theme === "dark" ? "plotly_dark" : "plotly_white",
            font: { color: styles.getPropertyValue("--text-color").trim() },
            plot_bgcolor: styles.getPropertyValue("--box-bg").trim(),
            paper_bgcolor: styles.getPropertyValue("--box-bg").trim(),
            xaxis: {
                gridcolor: styles.getPropertyValue("--grid-color").trim(),
                color: styles.getPropertyValue("--text-color").trim(),
                title: "Frequency (Hz)"
            },
            yaxis: {
                gridcolor: styles.getPropertyValue("--grid-color").trim(),
                color: styles.getPropertyValue("--text-color").trim()
            },
            margin: { t: 50, l: 50, r: 50, b: 50 }
        };
    }

    // Функция для очистки данных (удаление NaN/Infinity)
    function cleanData(arr) {
        if (!Array.isArray(arr)) return [];
        return arr.map(v => (typeof v === 'number' && isFinite(v)) ? v : null);
    }

    // Функция для создания графиков
    function drawPlots() {
        const theme = body.classList.contains("dark") ? "dark" : "light";
        const container = document.getElementById("plot-container");
        
        if (!container) {
            console.error("Plot container not found");
            return;
        }
        
        container.innerHTML = "";
        
        // 1. Рисуем основные графики (transfer_function и isolation_efficiency)
        try {
            if (mainPlots?.transfer_function?.x?.length > 0) {
                const mainDiv = document.createElement("div");
                mainDiv.className = "plot";
                mainDiv.id = "main-plot";
                container.appendChild(mainDiv);

                const transferData = mainPlots.transfer_function.x.map((xArr, i) => ({
                    x: cleanData(xArr),
                    y: cleanData(mainPlots.transfer_function.y[i]),
                    name: mainPlots.transfer_function.names[i] || `Line ${i+1}`,
                    mode: 'lines',
                    type: 'scatter'
                }));

                Plotly.newPlot(mainDiv.id, transferData, {
                    title: "Transfer Function",
                    ...getPlotlyLayoutOverrides(theme)
                });
            }

            if (mainPlots?.isolation_efficiency?.x?.length > 0) {
                const efficiencyDiv = document.createElement("div");
                efficiencyDiv.className = "plot";
                efficiencyDiv.id = "efficiency-plot";
                container.appendChild(efficiencyDiv);

                const efficiencyData = mainPlots.isolation_efficiency.x.map((xArr, i) => ({
                    x: cleanData(xArr),
                    y: cleanData(mainPlots.isolation_efficiency.y[i]),
                    name: mainPlots.isolation_efficiency.names[i] || `Line ${i+1}`,
                    mode: 'lines',
                    type: 'scatter'
                }));

                Plotly.newPlot(efficiencyDiv.id, efficiencyData, {
                    title: "Isolation Efficiency",
                    yaxis: { title: "Efficiency (dB)" },
                    ...getPlotlyLayoutOverrides(theme)
                });
            }
        } catch (e) {
            console.error("Error drawing main plots:", e);
        }

        // 2. Рисуем индивидуальные графики для каждого набора данных
        if (Array.isArray(datasets)) {
            datasets.forEach((dataset, index) => {
                try {
                    if (!dataset?.transfer_function?.x) return;

                    const div = document.createElement("div");
                    div.className = "plot";
                    div.id = `dataset-plot-${index}`;
                    container.appendChild(div);

                    // Подготовка данных для графика
                    const plotData = [];
                    
                    // Добавляем основные данные
                    if (dataset.transfer_function) {
                        plotData.push({
                            x: cleanData(dataset.transfer_function.x),
                            y: cleanData(dataset.transfer_function.y),
                            name: dataset.transfer_function.name || "Transfer Function",
                            mode: 'lines',
                            line: { color: '#1f77b4' }
                        });
                    }

                    // Добавляем сглаженные данные
                    if (dataset.smoothed_transfer) {
                        plotData.push({
                            x: cleanData(dataset.smoothed_transfer.x),
                            y: cleanData(dataset.smoothed_transfer.y),
                            name: dataset.smoothed_transfer.name || "Smoothed",
                            mode: 'lines',
                            line: { color: '#ff7f0e' }
                        });
                    }

                    // Добавляем пики, если есть
                    if (dataset.peaks?.length > 0) {
                        dataset.peaks.forEach(peak => {
                            plotData.push({
                                x: [peak.position[0]],
                                y: [peak.position[1]],
                                name: `Peak at ${peak.frequency.toFixed(2)} Hz`,
                                mode: 'markers',
                                marker: {
                                    size: 12,
                                    color: 'red'
                                }
                            });
                        });
                    }

                    // Настройки графика
                    const layout = {
                        title: dataset.title || `Dataset ${index + 1}`,
                        ...getPlotlyLayoutOverrides(theme),
                        showlegend: true,
                        hovermode: 'closest'
                    };

                    // Добавляем аннотации для пиков
                    if (dataset.peaks?.length > 0) {
                        layout.annotations = dataset.peaks.map(peak => ({
                            x: peak.position[0],
                            y: peak.position[1],
                            text: `Peak: ${peak.frequency.toFixed(2)} Hz`,
                            showarrow: true,
                            arrowhead: 7,
                            ax: 0,
                            ay: -40,
                            bgcolor: 'rgba(255,255,255,0.8)',
                            bordercolor: 'rgba(0,0,0,0.5)'
                        }));
                    }

                    Plotly.newPlot(div.id, plotData, layout);
                } catch (e) {
                    console.error(`Error drawing dataset ${index}:`, e);
                }
            });
        }

        // 3. Рисуем сводные данные (если есть)
        if (summaryData?.loads?.length > 0) {
            try {
                const summaryDiv = document.createElement("div");
                summaryDiv.className = "plot";
                summaryDiv.id = "summary-plot";
                container.appendChild(summaryDiv);

                const trace1 = {
                    x: summaryData.loads,
                    y: summaryData.dynamic_modules,
                    name: "Dynamic Modules",
                    mode: 'lines+markers',
                    type: 'scatter',
                    yaxis: 'y1'
                };

                const trace2 = {
                    x: summaryData.loads,
                    y: summaryData.damping_coeffs,
                    name: "Damping Coefficients",
                    mode: 'lines+markers',
                    type: 'scatter',
                    yaxis: 'y2'
                };

                Plotly.newPlot(summaryDiv.id, [trace1, trace2], {
                    title: "Summary Data",
                    ...getPlotlyLayoutOverrides(theme),
                    yaxis: { title: "Dynamic Modules (MPa)" },
                    yaxis2: {
                        title: "Damping Coefficients",
                        overlaying: 'y',
                        side: 'right'
                    }
                });
            } catch (e) {
                console.error("Error drawing summary plot:", e);
            }
        }
    }

    // Функция для переключения темы
    function applyTheme(theme) {
        if (theme === "dark") {
            body.classList.add("dark");
            if (themeToggle) themeToggle.textContent = "☀️ Light Theme";
        } else {
            body.classList.remove("dark");
            if (themeToggle) themeToggle.textContent = "🌙 Dark Theme";
        }
        drawPlots();
    }

    // Инициализация темы
    const savedTheme = localStorage.getItem("theme") || "dark";
    applyTheme(savedTheme);

    // Обработчик переключения темы
    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const newTheme = body.classList.contains("dark") ? "light" : "dark";
            localStorage.setItem("theme", newTheme);
            applyTheme(newTheme);
        });
    }

    // Первоначальная отрисовка графиков
    drawPlots();

    // Обработчик изменения размера окна
    window.addEventListener('resize', function() {
        Plotly.Plots.resize(document.getElementById("plot-container"));
    });
});