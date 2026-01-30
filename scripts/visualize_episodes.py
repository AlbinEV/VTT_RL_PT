#!/usr/bin/env python3
"""
Create interactive HTML visualization with PER-EPISODE view.
Shows 3D trajectory, force profiles, impedance parameters for each episode.
"""
import _path_setup

import argparse
import os
import glob
import h5py
import numpy as np
from pathlib import Path
import json

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Fixed Mode - Per Episode Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e8e8e8;
            min-height: 100vh;
        }}
        .header {{
            background: rgba(0,0,0,0.3);
            padding: 20px 40px;
            border-bottom: 2px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            font-size: 24px;
            color: #00d9ff;
        }}
        .header .subtitle {{
            color: #888;
            font-size: 13px;
        }}
        .episode-selector {{
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .selector-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .episode-selector label {{
            color: #aaa;
            font-size: 14px;
        }}
        .episode-selector select {{
            padding: 8px 15px;
            border-radius: 6px;
            border: 1px solid #0f3460;
            background: #1a1a2e;
            color: #00d9ff;
            font-size: 14px;
            cursor: pointer;
        }}
        .toggle {{
            display: flex;
            align-items: center;
            gap: 6px;
            color: #aaa;
            font-size: 13px;
        }}
        .toggle input {{
            accent-color: #00d9ff;
        }}
        .compare-selectors {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .hidden {{
            display: none;
        }}
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }}
        .stats-section {{
            margin-bottom: 12px;
        }}
        .stats-title {{
            color: #aaa;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin: 4px 0 8px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 12px 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-card .label {{
            color: #888;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .stat-card .value {{
            font-size: 20px;
            font-weight: bold;
            color: #00d9ff;
            margin-top: 3px;
        }}
        .stat-card .unit {{
            font-size: 12px;
            color: #666;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }}
        .plot-card {{
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .plot-card.full-width {{
            grid-column: span 2;
        }}
        .plot-card h3 {{
            color: #00d9ff;
            font-size: 14px;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .waypoint-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
        }}
        .waypoint-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            font-size: 11px;
        }}
        .waypoint-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        .impedance-info {{
            background: rgba(0,100,200,0.1);
            border: 1px solid rgba(0,150,255,0.3);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .impedance-info h4 {{
            color: #00d9ff;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        .impedance-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 10px;
        }}
        .impedance-cell {{
            text-align: center;
            padding: 8px;
            background: rgba(0,0,0,0.2);
            border-radius: 4px;
        }}
        .impedance-cell .axis {{
            font-size: 10px;
            color: #888;
        }}
        .impedance-cell .kp {{
            font-size: 16px;
            color: #4ecdc4;
            font-weight: bold;
        }}
        .impedance-cell .zeta {{
            font-size: 12px;
            color: #a855f7;
        }}
        @media (max-width: 1200px) {{
            .plot-grid {{ grid-template-columns: 1fr; }}
            .plot-card.full-width {{ grid-column: span 1; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🤖 Fixed Mode - Per Episode Analysis</h1>
        <div id="subtitle" class="subtitle">{controller_name} | {num_episodes} episodes | {total_steps} total steps</div>
        </div>
        <div class="episode-selector">
            <div class="selector-group">
                <label>Dataset:</label>
                <select id="dataset-select" onchange="onDatasetChange()"></select>
                <label>Episode:</label>
                <select id="episode-select" onchange="updatePlots()"></select>
            </div>
            <label class="toggle">
                <input type="checkbox" id="compare-toggle" onchange="onCompareToggle()" />
                Overlay
            </label>
            <div id="compare-selectors" class="compare-selectors hidden">
                <div class="selector-group">
                    <label>Compare B:</label>
                    <select id="compare-dataset-select" onchange="onCompareDatasetChange()"></select>
                    <label>Episode:</label>
                    <select id="compare-episode-select" onchange="updatePlots()"></select>
                </div>
                <div id="compare-selectors-2" class="selector-group">
                    <label>Compare C:</label>
                    <select id="compare-dataset-select-2" onchange="onCompareDatasetChange2()"></select>
                    <label>Episode:</label>
                    <select id="compare-episode-select-2" onchange="updatePlots()"></select>
                </div>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="impedance-info">
            <h4>🔧 OSC Impedance Parameters (Nominal)</h4>
            <div class="impedance-grid">
                <div class="impedance-cell">
                    <div class="axis">X</div>
                    <div class="kp">3500</div>
                    <div class="zeta">ζ=1.0</div>
                </div>
                <div class="impedance-cell">
                    <div class="axis">Y</div>
                    <div class="kp">1900</div>
                    <div class="zeta">ζ=1.0</div>
                </div>
                <div class="impedance-cell">
                    <div class="axis">Z</div>
                    <div class="kp">3500</div>
                    <div class="zeta">ζ=1.0</div>
                </div>
                <div class="impedance-cell">
                    <div class="axis">Rx</div>
                    <div class="kp">460</div>
                    <div class="zeta">ζ=1.0</div>
                </div>
                <div class="impedance-cell">
                    <div class="axis">Ry</div>
                    <div class="kp">460</div>
                    <div class="zeta">ζ=1.0</div>
                </div>
                <div class="impedance-cell">
                    <div class="axis">Rz</div>
                    <div class="kp">410</div>
                    <div class="zeta">ζ=1.0</div>
                </div>
            </div>
        </div>
        
        <div id="stats-container"></div>
        
        <div class="plot-grid">
            <div class="plot-card full-width">
                <h3>📍 3D End-Effector Trajectory</h3>
                <div id="plot-3d"></div>
                <div class="waypoint-legend">
                    <div class="waypoint-item"><div class="waypoint-dot" style="background:#e74c3c"></div>HOME (0)</div>
                    <div class="waypoint-item"><div class="waypoint-dot" style="background:#e67e22"></div>INTERMEDIATE (1)</div>
                    <div class="waypoint-item"><div class="waypoint-dot" style="background:#f1c40f"></div>APPROACH (2)</div>
                    <div class="waypoint-item"><div class="waypoint-dot" style="background:#2ecc71"></div>CONTACT (3)</div>
                    <div class="waypoint-item"><div class="waypoint-dot" style="background:#1abc9c"></div>LATERAL_0 (4)</div>
                    <div class="waypoint-item"><div class="waypoint-dot" style="background:#3498db"></div>LATERAL_1 (5)</div>
                    <div class="waypoint-item"><div class="waypoint-dot" style="background:#9b59b6"></div>LATERAL_2 (6)</div>
                    <div class="waypoint-item"><div class="waypoint-dot" style="background:#e91e63"></div>RISE (7)</div>
                </div>
            </div>
            
            <div class="plot-card">
                <h3>📊 EE Position (XYZ)</h3>
                <div id="plot-pos"></div>
            </div>
            
            <div class="plot-card">
                <h3>⚡ Contact Force Fz</h3>
                <div id="plot-force"></div>
            </div>
            
            <div class="plot-card">
                <h3>🔧 Stiffness Kp (all axes)</h3>
                <div id="plot-kp"></div>
            </div>
            
            <div class="plot-card">
                <h3>🌊 Damping ζ (all axes)</h3>
                <div id="plot-zeta"></div>
            </div>
            
            <div class="plot-card full-width">
                <h3>🎯 Waypoint Index</h3>
                <div id="plot-waypoint"></div>
            </div>
            
            <div class="plot-card full-width">
                <h3>💰 Reward per Step</h3>
                <div id="plot-reward"></div>
            </div>
        </div>
    </div>
    
    <script>
        const episodes = {episodes_json};
        
        const colors = {{
            primary: '#00d9ff',
            secondary: '#ff6b6b',
            tertiary: '#4ecdc4',
            quaternary: '#a855f7',
            grid: 'rgba(255,255,255,0.1)',
            paper: 'rgba(0,0,0,0)',
            plot: 'rgba(0,0,0,0.2)'
        }};
        
        const waypointColors = [
            '#e74c3c', '#e67e22', '#f1c40f', '#2ecc71',
            '#1abc9c', '#3498db', '#9b59b6', '#e91e63'
        ];
        
        const defaultLayout = {{
            paper_bgcolor: colors.paper,
            plot_bgcolor: colors.plot,
            font: {{ color: '#e8e8e8', size: 10 }},
            margin: {{ l: 45, r: 25, t: 25, b: 35 }},
            xaxis: {{
                gridcolor: colors.grid,
                zerolinecolor: colors.grid,
                title: {{ text: 'Step', standoff: 8 }}
            }},
            yaxis: {{
                gridcolor: colors.grid,
                zerolinecolor: colors.grid
            }},
            showlegend: true,
            legend: {{ x: 1, y: 1, xanchor: 'right', bgcolor: 'rgba(0,0,0,0.3)', font: {{size: 9}} }}
        }};
        
        const waypoints = [
            [0.55, 0.0, 0.55],
            [0.55, 0.0, 0.325],
            [0.55, 0.0, 0.1],
            [0.55, 0.0, 0.0],
            [0.55, -0.2, 0.0],
            [0.55, 0.0, 0.0],
            [0.55, 0.2, 0.0],
            [0.55, 0.0, 0.15]
        ];
        const waypointLabels = ['HOME', 'INTERM', 'APPROACH', 'CONTACT', 'LAT_0', 'LAT_1', 'LAT_2', 'RISE'];
        
        const datasetSelect = document.getElementById('dataset-select');
        const episodeSelect = document.getElementById('episode-select');
        const compareToggle = document.getElementById('compare-toggle');
        const compareSelectors = document.getElementById('compare-selectors');
        const compareDatasetSelect = document.getElementById('compare-dataset-select');
        const compareEpisodeSelect = document.getElementById('compare-episode-select');
        const compareSelectors2 = document.getElementById('compare-selectors-2');
        const compareDatasetSelect2 = document.getElementById('compare-dataset-select-2');
        const compareEpisodeSelect2 = document.getElementById('compare-episode-select-2');
        const statsContainer = document.getElementById('stats-container');
        const subtitle = document.getElementById('subtitle');
        let currentEpisodesA = [];
        let currentEpisodesB = [];
        let currentEpisodesC = [];
        let hasThirdDataset = false;

        function getDatasetName(ep) {{
            return ep.dataset || 'OSC';
        }}

        function getEpisodesByDataset(dataset) {{
            return episodes.filter((ep) => getDatasetName(ep) === dataset);
        }}

        function updateEpisodeOptions(selectEl, list) {{
            selectEl.innerHTML = list.map((ep, i) => {{
                const steps = ep.fz ? ep.fz.length : 0;
                const label = ep.filename ? ` - ${{ep.filename}}` : '';
                return `<option value="${{i}}">Episode ${{i}}${{label}} (${{steps}} steps)</option>`;
            }}).join('\\n');
            selectEl.value = "0";
        }}

        function updatePrimaryOptions() {{
            currentEpisodesA = getEpisodesByDataset(datasetSelect.value);
            updateEpisodeOptions(episodeSelect, currentEpisodesA);
        }}

        function updateCompareOptions() {{
            currentEpisodesB = getEpisodesByDataset(compareDatasetSelect.value);
            updateEpisodeOptions(compareEpisodeSelect, currentEpisodesB);
        }}

        function updateCompareOptions2() {{
            currentEpisodesC = getEpisodesByDataset(compareDatasetSelect2.value);
            updateEpisodeOptions(compareEpisodeSelect2, currentEpisodesC);
        }}

        function updateSubtitle() {{
            const datasetA = datasetSelect.value;
            const datasetB = compareToggle.checked ? compareDatasetSelect.value : null;
            const datasetC = compareToggle.checked ? compareDatasetSelect2.value : null;
            const totalStepsA = currentEpisodesA.reduce((sum, ep) => sum + (ep.fz ? ep.fz.length : 0), 0);
            if (datasetB) {{
                const totalStepsB = currentEpisodesB.reduce((sum, ep) => sum + (ep.fz ? ep.fz.length : 0), 0);
                if (hasThirdDataset && datasetC) {{
                    const totalStepsC = currentEpisodesC.reduce((sum, ep) => sum + (ep.fz ? ep.fz.length : 0), 0);
                    subtitle.textContent = `${{datasetA}} vs ${{datasetB}} vs ${{datasetC}} | ` +
                        `${{currentEpisodesA.length}}/${{currentEpisodesB.length}}/${{currentEpisodesC.length}} episodes | ` +
                        `${{totalStepsA}}/${{totalStepsB}}/${{totalStepsC}} steps`;
                }} else {{
                    subtitle.textContent = `${{datasetA}} vs ${{datasetB}} | ` +
                        `${{currentEpisodesA.length}}/${{currentEpisodesB.length}} episodes | ` +
                        `${{totalStepsA}}/${{totalStepsB}} steps`;
                }}
            }} else {{
                subtitle.textContent = `${{datasetA}} | ${{currentEpisodesA.length}} episodes | ${{totalStepsA}} total steps`;
            }}
        }}

        function onDatasetChange() {{
            updatePrimaryOptions();
            updateSubtitle();
            updatePlots();
        }}

        function onCompareDatasetChange() {{
            updateCompareOptions();
            updateSubtitle();
            updatePlots();
        }}

        function onCompareDatasetChange2() {{
            updateCompareOptions2();
            updateSubtitle();
            updatePlots();
        }}

        function onCompareToggle() {{
            const show = compareToggle.checked;
            compareSelectors.classList.toggle('hidden', !show);
            compareSelectors2.classList.toggle('hidden', !(show && hasThirdDataset));
            updateSubtitle();
            updatePlots();
        }}

        function initSelectors() {{
            const datasets = Array.from(new Set(episodes.map(getDatasetName)));
            datasetSelect.innerHTML = datasets.map((name) => `<option value="${{name}}">${{name}}</option>`).join('\\n');
            compareDatasetSelect.innerHTML = datasetSelect.innerHTML;
            compareDatasetSelect2.innerHTML = datasetSelect.innerHTML;

            const defaultPrimary = datasets.includes('OSC') ? 'OSC' : (datasets[0] || '');
            datasetSelect.value = defaultPrimary;
            const remaining = datasets.filter((name) => name !== defaultPrimary);
            compareDatasetSelect.value = remaining[0] || defaultPrimary;
            compareDatasetSelect2.value = remaining[1] || remaining[0] || defaultPrimary;

            compareToggle.disabled = datasets.length < 2;
            hasThirdDataset = datasets.length >= 3;
            if (datasets.length < 2) {{
                compareToggle.checked = false;
                compareSelectors.classList.add('hidden');
            }}
            compareDatasetSelect2.disabled = !hasThirdDataset;
            compareEpisodeSelect2.disabled = !hasThirdDataset;
            compareSelectors2.classList.toggle('hidden', !hasThirdDataset);

            updatePrimaryOptions();
            updateCompareOptions();
            updateCompareOptions2();
            updateSubtitle();
            updatePlots();
        }}

        function buildStatsBlock(ep, label) {{
            if (!ep || !ep.fz || !ep.reward) {{
                return '';
            }}
            const contactSteps = ep.fz.filter(f => Math.abs(f) > 0.5).length;
            const contactPct = (100 * contactSteps / ep.fz.length).toFixed(0);
            const meanFz = ep.fz.filter(f => Math.abs(f) > 0.5).reduce((a,b) => a + Math.abs(b), 0) / Math.max(contactSteps, 1);
            const cumReward = ep.reward.reduce((a, b) => a + b, 0);
            const waypointCount = ep.wpt_idx ? new Set(ep.wpt_idx).size : 0;
            return `
                <div class="stats-section">
                    <div class="stats-title">${{label}}</div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="label">Steps</div>
                            <div class="value">${{ep.fz.length}}</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Duration</div>
                            <div class="value">${{(ep.fz.length * 0.1).toFixed(1)}} <span class="unit">s</span></div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Waypoints</div>
                            <div class="value">${{waypointCount}}</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Contact Steps</div>
                            <div class="value">${{contactSteps}} <span class="unit">(${{contactPct}}%)</span></div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Mean |Fz|</div>
                            <div class="value">${{meanFz.toFixed(2)}} <span class="unit">N</span></div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Total Reward</div>
                            <div class="value">${{cumReward.toFixed(1)}}</div>
                        </div>
                    </div>
                </div>
            `;
        }}

        function updatePlots() {{
            if (!currentEpisodesA.length) {{
                return;
            }}
            const epIdxA = parseInt(episodeSelect.value);
            const epA = currentEpisodesA[epIdxA];
            if (!epA) {{
                return;
            }}

            const compare = compareToggle.checked && currentEpisodesB.length > 0;
            const compareC = compare && currentEpisodesC.length > 0 && !compareEpisodeSelect2.disabled;
            const epIdxB = parseInt(compareEpisodeSelect.value);
            const epIdxC = parseInt(compareEpisodeSelect2.value);
            const epB = compare ? currentEpisodesB[epIdxB] : null;
            const epC = compareC ? currentEpisodesC[epIdxC] : null;

            statsContainer.innerHTML = buildStatsBlock(epA, datasetSelect.value) +
                (compare && epB ? buildStatsBlock(epB, compareDatasetSelect.value) : '') +
                (compareC && epC ? buildStatsBlock(epC, compareDatasetSelect2.value) : '');

            // 3D Trajectory
            const trace3dA = {{
                type: 'scatter3d',
                mode: 'lines+markers',
                x: epA.ee_pos_x,
                y: epA.ee_pos_y,
                z: epA.ee_pos_z,
                marker: {{
                    size: 2,
                    color: epA.wpt_idx,
                    colorscale: waypointColors.map((c, i) => [i/7, c]).concat([[1, waypointColors[7]]]),
                    cmin: 0,
                    cmax: 7
                }},
                line: {{ width: 1.5, color: colors.primary }},
                name: datasetSelect.value
            }};
            const traces3d = [trace3dA];
            if (compare && epB) {{
                traces3d.push({{
                    type: 'scatter3d',
                    mode: 'lines',
                    x: epB.ee_pos_x,
                    y: epB.ee_pos_y,
                    z: epB.ee_pos_z,
                    line: {{ width: 1.5, color: colors.secondary }},
                    name: compareDatasetSelect.value
                }});
            }}
            if (compareC && epC) {{
                traces3d.push({{
                    type: 'scatter3d',
                    mode: 'lines',
                    x: epC.ee_pos_x,
                    y: epC.ee_pos_y,
                    z: epC.ee_pos_z,
                    line: {{ width: 1.5, color: colors.tertiary }},
                    name: compareDatasetSelect2.value
                }});
            }}

            const waypointMarkers = {{
                type: 'scatter3d',
                mode: 'markers+text',
                x: waypoints.map(w => w[0]),
                y: waypoints.map(w => w[1]),
                z: waypoints.map(w => w[2]),
                marker: {{ size: 8, color: waypointColors, symbol: 'diamond' }},
                text: waypointLabels,
                textposition: 'top center',
                textfont: {{ size: 9, color: '#fff' }},
                name: 'Waypoints'
            }};
            traces3d.push(waypointMarkers);

            Plotly.react('plot-3d', traces3d, {{
                ...defaultLayout,
                height: 400,
                scene: {{
                    xaxis: {{ title: 'X', gridcolor: colors.grid }},
                    yaxis: {{ title: 'Y', gridcolor: colors.grid }},
                    zaxis: {{ title: 'Z', gridcolor: colors.grid }},
                    camera: {{ eye: {{ x: 1.5, y: 1.5, z: 1.2 }} }},
                    aspectmode: 'data'
                }}
            }});

            // Position
            const posTraces = [
                {{ y: epA.ee_pos_x, name: `${{datasetSelect.value}} X`, line: {{ color: '#e74c3c', width: 1 }} }},
                {{ y: epA.ee_pos_y, name: `${{datasetSelect.value}} Y`, line: {{ color: '#2ecc71', width: 1 }} }},
                {{ y: epA.ee_pos_z, name: `${{datasetSelect.value}} Z`, line: {{ color: '#3498db', width: 1 }} }}
            ];
            if (compare && epB) {{
                posTraces.push({{ y: epB.ee_pos_x, name: `${{compareDatasetSelect.value}} X`, line: {{ color: '#e74c3c', width: 1, dash: 'dot' }} }});
                posTraces.push({{ y: epB.ee_pos_y, name: `${{compareDatasetSelect.value}} Y`, line: {{ color: '#2ecc71', width: 1, dash: 'dot' }} }});
                posTraces.push({{ y: epB.ee_pos_z, name: `${{compareDatasetSelect.value}} Z`, line: {{ color: '#3498db', width: 1, dash: 'dot' }} }});
            }}
            if (compareC && epC) {{
                posTraces.push({{ y: epC.ee_pos_x, name: `${{compareDatasetSelect2.value}} X`, line: {{ color: '#e74c3c', width: 1, dash: 'dash' }} }});
                posTraces.push({{ y: epC.ee_pos_y, name: `${{compareDatasetSelect2.value}} Y`, line: {{ color: '#2ecc71', width: 1, dash: 'dash' }} }});
                posTraces.push({{ y: epC.ee_pos_z, name: `${{compareDatasetSelect2.value}} Z`, line: {{ color: '#3498db', width: 1, dash: 'dash' }} }});
            }}
            Plotly.react('plot-pos', posTraces, {{ ...defaultLayout, height: 220, yaxis: {{ ...defaultLayout.yaxis, title: 'Position (m)' }} }});

            // Force
            const targetLen = Math.max(epA.fz.length, epB ? epB.fz.length : 0, epC ? epC.fz.length : 0);
            const forceTraces = [
                {{ y: epA.fz, name: `${{datasetSelect.value}} Fz`, line: {{ color: colors.primary }}, fill: 'tozeroy' }}
            ];
            if (compare && epB) {{
                forceTraces.push({{ y: epB.fz, name: `${{compareDatasetSelect.value}} Fz`, line: {{ color: colors.secondary, dash: 'dot' }} }});
            }}
            if (compareC && epC) {{
                forceTraces.push({{ y: epC.fz, name: `${{compareDatasetSelect2.value}} Fz`, line: {{ color: colors.tertiary, dash: 'dash' }} }});
            }}
            forceTraces.push({{ y: Array.from({{length: targetLen}}, () => -2), name: 'Target', line: {{ color: '#fff', dash: 'dash', width: 1 }} }});
            Plotly.react('plot-force', forceTraces, {{ ...defaultLayout, height: 220, yaxis: {{ ...defaultLayout.yaxis, title: 'Force (N)' }} }});

            // Kp
            const kpTraces = [];
            if (epA.kp_x) {{
                kpTraces.push({{ y: epA.kp_x, name: `${{datasetSelect.value}} Kp_x`, line: {{ color: '#e74c3c', width: 1 }} }});
                kpTraces.push({{ y: epA.kp_y, name: `${{datasetSelect.value}} Kp_y`, line: {{ color: '#2ecc71', width: 1 }} }});
                kpTraces.push({{ y: epA.kp_z, name: `${{datasetSelect.value}} Kp_z`, line: {{ color: '#3498db', width: 1.5 }} }});
            }}
            if (compare && epB && epB.kp_x) {{
                kpTraces.push({{ y: epB.kp_x, name: `${{compareDatasetSelect.value}} Kp_x`, line: {{ color: '#e74c3c', width: 1, dash: 'dot' }} }});
                kpTraces.push({{ y: epB.kp_y, name: `${{compareDatasetSelect.value}} Kp_y`, line: {{ color: '#2ecc71', width: 1, dash: 'dot' }} }});
                kpTraces.push({{ y: epB.kp_z, name: `${{compareDatasetSelect.value}} Kp_z`, line: {{ color: '#3498db', width: 1.5, dash: 'dot' }} }});
            }}
            if (compareC && epC && epC.kp_x) {{
                kpTraces.push({{ y: epC.kp_x, name: `${{compareDatasetSelect2.value}} Kp_x`, line: {{ color: '#e74c3c', width: 1, dash: 'dash' }} }});
                kpTraces.push({{ y: epC.kp_y, name: `${{compareDatasetSelect2.value}} Kp_y`, line: {{ color: '#2ecc71', width: 1, dash: 'dash' }} }});
                kpTraces.push({{ y: epC.kp_z, name: `${{compareDatasetSelect2.value}} Kp_z`, line: {{ color: '#3498db', width: 1.5, dash: 'dash' }} }});
            }}
            Plotly.react('plot-kp', kpTraces, {{ ...defaultLayout, height: 220, yaxis: {{ ...defaultLayout.yaxis, title: 'Stiffness (N/m)' }} }});

            // Zeta
            const zetaTraces = [];
            if (epA.zeta_x) {{
                zetaTraces.push({{ y: epA.zeta_x, name: `${{datasetSelect.value}} ζ_x`, line: {{ color: '#e74c3c', width: 1 }} }});
                zetaTraces.push({{ y: epA.zeta_y, name: `${{datasetSelect.value}} ζ_y`, line: {{ color: '#2ecc71', width: 1 }} }});
                zetaTraces.push({{ y: epA.zeta_z, name: `${{datasetSelect.value}} ζ_z`, line: {{ color: '#3498db', width: 1.5 }} }});
            }}
            if (compare && epB && epB.zeta_x) {{
                zetaTraces.push({{ y: epB.zeta_x, name: `${{compareDatasetSelect.value}} ζ_x`, line: {{ color: '#e74c3c', width: 1, dash: 'dot' }} }});
                zetaTraces.push({{ y: epB.zeta_y, name: `${{compareDatasetSelect.value}} ζ_y`, line: {{ color: '#2ecc71', width: 1, dash: 'dot' }} }});
                zetaTraces.push({{ y: epB.zeta_z, name: `${{compareDatasetSelect.value}} ζ_z`, line: {{ color: '#3498db', width: 1.5, dash: 'dot' }} }});
            }}
            if (compareC && epC && epC.zeta_x) {{
                zetaTraces.push({{ y: epC.zeta_x, name: `${{compareDatasetSelect2.value}} ζ_x`, line: {{ color: '#e74c3c', width: 1, dash: 'dash' }} }});
                zetaTraces.push({{ y: epC.zeta_y, name: `${{compareDatasetSelect2.value}} ζ_y`, line: {{ color: '#2ecc71', width: 1, dash: 'dash' }} }});
                zetaTraces.push({{ y: epC.zeta_z, name: `${{compareDatasetSelect2.value}} ζ_z`, line: {{ color: '#3498db', width: 1.5, dash: 'dash' }} }});
            }}
            Plotly.react('plot-zeta', zetaTraces, {{ ...defaultLayout, height: 220, yaxis: {{ ...defaultLayout.yaxis, title: 'Damping Ratio' }} }});

            // Waypoint
            const waypointTraces = [
                {{ y: epA.wpt_idx, name: `${{datasetSelect.value}} Waypoint`, line: {{ color: colors.primary }}, fill: 'tozeroy' }}
            ];
            if (compare && epB) {{
                waypointTraces.push({{ y: epB.wpt_idx, name: `${{compareDatasetSelect.value}} Waypoint`, line: {{ color: colors.secondary, dash: 'dot' }} }});
            }}
            if (compareC && epC) {{
                waypointTraces.push({{ y: epC.wpt_idx, name: `${{compareDatasetSelect2.value}} Waypoint`, line: {{ color: colors.tertiary, dash: 'dash' }} }});
            }}
            Plotly.react('plot-waypoint', waypointTraces, {{ ...defaultLayout, height: 180, yaxis: {{ ...defaultLayout.yaxis, title: 'Waypoint', range: [-0.5, 8] }} }});

            // Reward
            const rewardTraces = [
                {{ y: epA.reward, name: `${{datasetSelect.value}} Reward`, line: {{ color: '#f1c40f' }}, fill: 'tozeroy' }}
            ];
            if (compare && epB) {{
                rewardTraces.push({{ y: epB.reward, name: `${{compareDatasetSelect.value}} Reward`, line: {{ color: '#f39c12', dash: 'dot' }} }});
            }}
            if (compareC && epC) {{
                rewardTraces.push({{ y: epC.reward, name: `${{compareDatasetSelect2.value}} Reward`, line: {{ color: '#f7b731', dash: 'dash' }} }});
            }}
            Plotly.react('plot-reward', rewardTraces, {{ ...defaultLayout, height: 180, yaxis: {{ ...defaultLayout.yaxis, title: 'Reward' }} }});
        }}
        
        // Initial plot
        initSelectors();
    </script>
</body>
</html>
"""


def load_episode_data(filepath: str) -> dict:
    """Load single episode data from H5 file."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            data[key] = f[key][:]
        # Get attributes
        data['_attrs'] = dict(f.attrs)
    return data


def _extract_positions(tcp_positions):
    if not tcp_positions:
        return [], [], []
    pos_x = []
    pos_y = []
    pos_z = []
    for pos in tcp_positions:
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            pos_x.append(float(pos[0]))
            pos_y.append(float(pos[1]))
            pos_z.append(float(pos[2]))
        else:
            pos_x.append(0.0)
            pos_y.append(0.0)
            pos_z.append(0.0)
    return pos_x, pos_y, pos_z


def _extract_fz(contact_forces):
    if not contact_forces:
        return []
    first = contact_forces[0]
    if isinstance(first, (list, tuple)) and len(first) >= 3:
        return [float(f[2]) if isinstance(f, (list, tuple)) and len(f) >= 3 else 0.0 for f in contact_forces]
    return [float(f) for f in contact_forces]


def load_trajectories_json(path: str, label=None):
    if not path:
        return []
    if os.path.isdir(path):
        path = os.path.join(path, "trajectories.json")
    if not os.path.isfile(path):
        print(f"[WARN] trajectories.json not found at {path}")
        return []
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        episodes_raw = payload.get("episodes", [])
    elif isinstance(payload, list):
        episodes_raw = payload
    else:
        episodes_raw = []

    episodes = []
    source_label = label or os.path.basename(os.path.dirname(path))
    for i, ep in enumerate(episodes_raw):
        tcp_positions = ep.get("tcp_positions", [])
        pos_x, pos_y, pos_z = _extract_positions(tcp_positions)
        fz = _extract_fz(ep.get("contact_forces", []))
        rewards = [float(r) for r in ep.get("rewards", [])]
        kp_z = [float(k) for k in ep.get("kp_z_values", [])]
        zeta_z = [float(z) for z in ep.get("zeta_z_values", [])]
        wpt_len = len(pos_x) or len(fz) or len(rewards)
        ep_data = {
            "idx": i,
            "filename": f"{source_label}/episode_{ep.get('episode', i)}.npz",
            "dataset": source_label,
            "ee_pos_x": pos_x,
            "ee_pos_y": pos_y,
            "ee_pos_z": pos_z,
            "fz": fz,
            "wpt_idx": [0] * wpt_len,
            "reward": rewards,
        }
        if kp_z:
            ep_data["kp_x"] = kp_z
            ep_data["kp_y"] = kp_z
            ep_data["kp_z"] = kp_z
        if zeta_z:
            ep_data["zeta_x"] = zeta_z
            ep_data["zeta_y"] = zeta_z
            ep_data["zeta_z"] = zeta_z
        episodes.append(ep_data)
    return episodes


def create_episode_visualization(
    data_dir: str,
    output_path: str,
    extra_json=None,
    extra_label=None,
    extra_json2=None,
    extra_label2=None,
):
    """Create HTML visualization with per-episode view."""
    
    # Find all episode files
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    extra_episodes = load_trajectories_json(extra_json, label=extra_label)
    extra_episodes_2 = load_trajectories_json(extra_json2, label=extra_label2)

    if not h5_files and not extra_episodes and not extra_episodes_2:
        print(f"No H5 files found in {data_dir} and no extra episodes provided")
        return

    if h5_files:
        print(f"Found {len(h5_files)} episode files")
    
    # Load all episodes
    episodes = []
    total_steps = 0
    
    for i, filepath in enumerate(h5_files):
        data = load_episode_data(filepath)
        
        ep_data = {
            'idx': i,
            'filename': os.path.basename(filepath),
            'dataset': 'OSC',
            'ee_pos_x': data['ee_pos'][:, 0].tolist() if 'ee_pos' in data else [],
            'ee_pos_y': data['ee_pos'][:, 1].tolist() if 'ee_pos' in data else [],
            'ee_pos_z': data['ee_pos'][:, 2].tolist() if 'ee_pos' in data else [],
            'fz': data['fz'].tolist() if 'fz' in data else [],
            'wpt_idx': data['wpt_idx'].tolist() if 'wpt_idx' in data else [],
            'reward': data['reward'].tolist() if 'reward' in data else [],
        }
        
        # Handle Kp and zeta (may have different shapes)
        if 'kp' in data and len(data['kp'].shape) > 1 and data['kp'].shape[1] >= 3:
            ep_data['kp_x'] = data['kp'][:, 0].tolist()
            ep_data['kp_y'] = data['kp'][:, 1].tolist()
            ep_data['kp_z'] = data['kp'][:, 2].tolist()
        elif 'kp' in data:
            ep_data['kp_z'] = data['kp'].flatten().tolist()
            ep_data['kp_x'] = ep_data['kp_z']
            ep_data['kp_y'] = ep_data['kp_z']
        
        if 'zeta' in data and len(data['zeta'].shape) > 1 and data['zeta'].shape[1] >= 3:
            ep_data['zeta_x'] = data['zeta'][:, 0].tolist()
            ep_data['zeta_y'] = data['zeta'][:, 1].tolist()
            ep_data['zeta_z'] = data['zeta'][:, 2].tolist()
        elif 'zeta' in data:
            ep_data['zeta_z'] = data['zeta'].flatten().tolist()
            ep_data['zeta_x'] = ep_data['zeta_z']
            ep_data['zeta_y'] = ep_data['zeta_z']
        
        episodes.append(ep_data)
        total_steps += len(ep_data['fz'])
        
        print(f"  Episode {i}: {len(ep_data['fz'])} steps, reward={sum(ep_data['reward']):.1f}")

    for j, ep_data in enumerate(extra_episodes):
        episodes.append(ep_data)
        ep_steps = len(ep_data.get("fz", [])) or len(ep_data.get("ee_pos_x", []))
        total_steps += ep_steps
        print(f"  Extra Episode {j}: {ep_steps} steps, reward={sum(ep_data.get('reward', [])):.1f}")

    for j, ep_data in enumerate(extra_episodes_2):
        episodes.append(ep_data)
        ep_steps = len(ep_data.get("fz", [])) or len(ep_data.get("ee_pos_x", []))
        total_steps += ep_steps
        print(f"  Extra Episode {j + len(extra_episodes)}: {ep_steps} steps, reward={sum(ep_data.get('reward', [])):.1f}")

    for i, ep in enumerate(episodes):
        ep["idx"] = i
    
    # Determine controller name from first episode
    controller_name = "OSC Pure" if 'OSC' in h5_files[0] or 'osc' in h5_files[0] else "Fixed Mode"
    
    # Generate HTML
    html = HTML_TEMPLATE.format(
        controller_name=controller_name,
        num_episodes=len(episodes),
        total_steps=total_steps,
        episodes_json=json.dumps(episodes)
    )
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"\n✅ Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create per-episode trajectory visualization")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing episode H5 files")
    parser.add_argument("--output", type=str, default="episode_visualization.html", help="Output HTML file")
    parser.add_argument("--extra_json", type=str, default=None, help="Optional trajectories.json path or directory")
    parser.add_argument("--extra_label", type=str, default=None, help="Label for extra trajectories dataset")
    parser.add_argument("--extra_json2", type=str, default=None, help="Second optional trajectories.json path or directory")
    parser.add_argument("--extra_label2", type=str, default=None, help="Label for second extra trajectories dataset")
    args = parser.parse_args()
    
    create_episode_visualization(
        args.input_dir,
        args.output,
        extra_json=args.extra_json,
        extra_label=args.extra_label,
        extra_json2=args.extra_json2,
        extra_label2=args.extra_label2,
    )


if __name__ == "__main__":
    main()
