#!/usr/bin/env python3
"""
Interactive HTML trajectory visualization for Fixed Mode data.
Similar to LfD discrete visualization.

Author: Albin Bajrami
"""
import _path_setup

import argparse
import h5py
import numpy as np
from pathlib import Path
import json


def create_html_visualization(data_files, output_file):
    """Create interactive HTML visualization with Plotly."""
    
    # Load all data
    all_runs = []
    for f in data_files:
        with h5py.File(f, "r") as hf:
            run_data = {
                "filename": f.name,
                "ee_pos": hf["ee_pos"][:],
                "fz": hf["fz"][:],
                "contact_force": hf["contact_force"][:],
                "kp": hf["kp"][:],
                "zeta": hf["zeta"][:],
                "reward": hf["reward"][:],
                "phase": hf["phase"][:],
                "sim_time": hf["sim_time"][:],
                "joint_pos": hf["joint_pos"][:],
                "action": hf["action"][:],
                "total_reward": hf.attrs["total_reward"],
                "contact_ratio": hf.attrs["contact_ratio"],
            }
            all_runs.append(run_data)
    
    # Convert numpy arrays to lists for JSON
    def to_list(arr):
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        return arr
    
    # Prepare data for JavaScript
    js_data = []
    for i, run in enumerate(all_runs):
        js_data.append({
            "name": f"Run {i}",
            "filename": run["filename"],
            "total_reward": float(run["total_reward"]),
            "contact_ratio": float(run["contact_ratio"]),
            "ee_x": to_list(run["ee_pos"][:, 0]),
            "ee_y": to_list(run["ee_pos"][:, 1]),
            "ee_z": to_list(run["ee_pos"][:, 2]),
            "fz": to_list(run["fz"]),
            "fx": to_list(run["contact_force"][:, 0]),
            "fy": to_list(run["contact_force"][:, 1]),
            "kp_z": to_list(run["kp"][:, 2]),  # Z axis
            "kp_rx": to_list(run["kp"][:, 3]),  # RX axis
            "zeta_z": to_list(run["zeta"][:, 2]),
            "zeta_rx": to_list(run["zeta"][:, 3]),
            "reward": to_list(run["reward"]),
            "phase": to_list(run["phase"]),
            "time": to_list(run["sim_time"]),
            "action": to_list(run["action"]),
        })
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Fixed Mode Polishing - Trajectory Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #00d9ff;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
        }}
        .subtitle {{
            color: #888;
            font-size: 14px;
        }}
        .container {{
            display: grid;
            grid-template-columns: 250px 1fr;
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }}
        .sidebar {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 15px;
            max-height: 80vh;
            overflow-y: auto;
        }}
        .sidebar h3 {{
            color: #00d9ff;
            margin-top: 0;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }}
        .run-item {{
            padding: 10px;
            margin: 5px 0;
            background: rgba(0, 217, 255, 0.1);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid transparent;
        }}
        .run-item:hover {{
            background: rgba(0, 217, 255, 0.2);
            border-color: #00d9ff;
        }}
        .run-item.active {{
            background: rgba(0, 217, 255, 0.3);
            border-color: #00d9ff;
        }}
        .run-stats {{
            font-size: 11px;
            color: #888;
            margin-top: 5px;
        }}
        .main-content {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .plot-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .plot-container {{
            background: rgba(255,255,255,0.02);
            border-radius: 12px;
            padding: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .plot-container.full-width {{
            grid-column: span 2;
        }}
        .stats-panel {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: rgba(0, 217, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            border: 1px solid rgba(0, 217, 255, 0.2);
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #00d9ff;
        }}
        .stat-label {{
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }}
        .negative {{ color: #ff6b6b; }}
        .positive {{ color: #51cf66; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Fixed Mode Polishing</h1>
        <div class="subtitle">Interactive Trajectory Visualization | {len(all_runs)} runs loaded</div>
    </div>
    
    <div class="stats-panel" id="stats-panel">
        <div class="stat-card">
            <div class="stat-value" id="stat-reward">-</div>
            <div class="stat-label">Total Reward</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="stat-fz">-</div>
            <div class="stat-label">Mean Fz [N]</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="stat-contact">-</div>
            <div class="stat-label">Contact Ratio</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="stat-steps">-</div>
            <div class="stat-label">Steps</div>
        </div>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <h3>📂 Runs</h3>
            <div id="run-list"></div>
        </div>
        
        <div class="main-content">
            <div class="plot-row">
                <div class="plot-container" id="plot-3d"></div>
                <div class="plot-container" id="plot-force"></div>
            </div>
            <div class="plot-row">
                <div class="plot-container" id="plot-impedance"></div>
                <div class="plot-container" id="plot-reward"></div>
            </div>
            <div class="plot-row">
                <div class="plot-container full-width" id="plot-actions"></div>
            </div>
        </div>
    </div>

    <script>
        const runData = {json.dumps(js_data)};
        let currentRun = 0;
        
        const darkLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0.2)',
            font: {{ color: '#eee', size: 11 }},
            margin: {{ t: 40, b: 40, l: 50, r: 20 }},
            xaxis: {{ gridcolor: 'rgba(255,255,255,0.1)', zerolinecolor: 'rgba(255,255,255,0.2)' }},
            yaxis: {{ gridcolor: 'rgba(255,255,255,0.1)', zerolinecolor: 'rgba(255,255,255,0.2)' }},
            legend: {{ bgcolor: 'rgba(0,0,0,0.3)', bordercolor: 'rgba(255,255,255,0.1)' }}
        }};
        
        function populateRunList() {{
            const list = document.getElementById('run-list');
            runData.forEach((run, i) => {{
                const item = document.createElement('div');
                item.className = 'run-item' + (i === 0 ? ' active' : '');
                item.innerHTML = `
                    <strong>${{run.name}}</strong>
                    <div class="run-stats">
                        Reward: ${{run.total_reward.toFixed(1)}} | 
                        Contact: ${{(run.contact_ratio * 100).toFixed(0)}}%
                    </div>
                `;
                item.onclick = () => selectRun(i);
                list.appendChild(item);
            }});
        }}
        
        function selectRun(idx) {{
            currentRun = idx;
            document.querySelectorAll('.run-item').forEach((el, i) => {{
                el.className = 'run-item' + (i === idx ? ' active' : '');
            }});
            updatePlots();
            updateStats();
        }}
        
        function updateStats() {{
            const run = runData[currentRun];
            const meanFz = run.fz.reduce((a, b) => a + b, 0) / run.fz.length;
            
            document.getElementById('stat-reward').textContent = run.total_reward.toFixed(1);
            document.getElementById('stat-fz').innerHTML = `<span class="${{meanFz < 0 ? 'negative' : ''}}">${{meanFz.toFixed(2)}}</span>`;
            document.getElementById('stat-contact').textContent = (run.contact_ratio * 100).toFixed(1) + '%';
            document.getElementById('stat-steps').textContent = run.fz.length;
        }}
        
        function updatePlots() {{
            const run = runData[currentRun];
            const time = run.time;
            
            // 3D Trajectory
            const trace3d = {{
                x: run.ee_x,
                y: run.ee_y,
                z: run.ee_z,
                mode: 'lines+markers',
                type: 'scatter3d',
                marker: {{
                    size: 3,
                    color: run.fz,
                    colorscale: 'RdBu',
                    colorbar: {{ title: 'Fz [N]', len: 0.5 }},
                    reversescale: true
                }},
                line: {{ width: 2 }},
                name: 'EE Trajectory'
            }};
            
            const layout3d = {{
                ...darkLayout,
                title: '3D End-Effector Trajectory',
                scene: {{
                    xaxis: {{ title: 'X [m]', gridcolor: 'rgba(255,255,255,0.1)' }},
                    yaxis: {{ title: 'Y [m]', gridcolor: 'rgba(255,255,255,0.1)' }},
                    zaxis: {{ title: 'Z [m]', gridcolor: 'rgba(255,255,255,0.1)' }},
                    bgcolor: 'rgba(0,0,0,0.2)',
                    aspectratio: {{ x: 1, y: 1, z: 0.5 }}
                }},
                height: 400
            }};
            
            Plotly.newPlot('plot-3d', [trace3d], layout3d);
            
            // Force plot
            const traceFz = {{
                x: time,
                y: run.fz,
                mode: 'lines',
                name: 'Fz',
                line: {{ color: '#ff6b6b', width: 2 }}
            }};
            const traceFx = {{
                x: time,
                y: run.fx,
                mode: 'lines',
                name: 'Fx',
                line: {{ color: '#4dabf7', width: 1 }},
                visible: 'legendonly'
            }};
            const traceFy = {{
                x: time,
                y: run.fy,
                mode: 'lines',
                name: 'Fy',
                line: {{ color: '#51cf66', width: 1 }},
                visible: 'legendonly'
            }};
            const targetLine = {{
                x: [time[0], time[time.length-1]],
                y: [-20, -20],
                mode: 'lines',
                name: 'Target (-20N)',
                line: {{ color: '#ffd43b', width: 2, dash: 'dash' }}
            }};
            
            const layoutForce = {{
                ...darkLayout,
                title: 'Contact Force',
                xaxis: {{ ...darkLayout.xaxis, title: 'Time [s]' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Force [N]' }},
                height: 400,
                showlegend: true
            }};
            
            Plotly.newPlot('plot-force', [traceFz, traceFx, traceFy, targetLine], layoutForce);
            
            // Impedance plot
            const traceKpZ = {{
                x: time,
                y: run.kp_z,
                mode: 'lines',
                name: 'Kp_z',
                line: {{ color: '#00d9ff', width: 2 }}
            }};
            const traceKpRx = {{
                x: time,
                y: run.kp_rx,
                mode: 'lines',
                name: 'Kp_rx',
                line: {{ color: '#845ef7', width: 2 }}
            }};
            const traceZetaZ = {{
                x: time,
                y: run.zeta_z,
                mode: 'lines',
                name: 'ζ_z',
                line: {{ color: '#ff922b', width: 2 }},
                yaxis: 'y2'
            }};
            const traceZetaRx = {{
                x: time,
                y: run.zeta_rx,
                mode: 'lines',
                name: 'ζ_rx',
                line: {{ color: '#f06595', width: 2 }},
                yaxis: 'y2'
            }};
            
            const layoutImp = {{
                ...darkLayout,
                title: 'Variable Impedance (Kp & ζ)',
                xaxis: {{ ...darkLayout.xaxis, title: 'Time [s]' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Kp [N/m or Nm/rad]', side: 'left' }},
                yaxis2: {{
                    title: 'Damping Ratio ζ',
                    overlaying: 'y',
                    side: 'right',
                    gridcolor: 'rgba(255,255,255,0.05)'
                }},
                height: 400,
                showlegend: true
            }};
            
            Plotly.newPlot('plot-impedance', [traceKpZ, traceKpRx, traceZetaZ, traceZetaRx], layoutImp);
            
            // Reward plot
            const traceReward = {{
                x: time,
                y: run.reward,
                mode: 'lines',
                name: 'Reward',
                fill: 'tozeroy',
                line: {{ color: '#51cf66', width: 2 }}
            }};
            const cumReward = [];
            let cumSum = 0;
            run.reward.forEach(r => {{
                cumSum += r;
                cumReward.push(cumSum);
            }});
            const traceCumReward = {{
                x: time,
                y: cumReward,
                mode: 'lines',
                name: 'Cumulative',
                line: {{ color: '#ffd43b', width: 2 }},
                yaxis: 'y2'
            }};
            
            const layoutReward = {{
                ...darkLayout,
                title: 'Reward',
                xaxis: {{ ...darkLayout.xaxis, title: 'Time [s]' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Step Reward' }},
                yaxis2: {{
                    title: 'Cumulative Reward',
                    overlaying: 'y',
                    side: 'right',
                    gridcolor: 'rgba(255,255,255,0.05)'
                }},
                height: 400,
                showlegend: true
            }};
            
            Plotly.newPlot('plot-reward', [traceReward, traceCumReward], layoutReward);
            
            // Actions plot
            const actionTraces = [];
            const actionNames = ['ΔKp_z', 'Δζ_z', 'ΔKp_rx', 'Δζ_rx'];
            const actionColors = ['#00d9ff', '#ff922b', '#845ef7', '#f06595'];
            for (let i = 0; i < 4; i++) {{
                actionTraces.push({{
                    x: time,
                    y: run.action.map(a => a[i]),
                    mode: 'lines',
                    name: actionNames[i],
                    line: {{ color: actionColors[i], width: 1.5 }}
                }});
            }}
            
            const layoutActions = {{
                ...darkLayout,
                title: 'Actions (Impedance Deltas)',
                xaxis: {{ ...darkLayout.xaxis, title: 'Time [s]' }},
                yaxis: {{ ...darkLayout.yaxis, title: 'Action [-1, 1]' }},
                height: 300,
                showlegend: true
            }};
            
            Plotly.newPlot('plot-actions', actionTraces, layoutActions);
        }}
        
        // Initialize
        populateRunList();
        updateStats();
        updatePlots();
    </script>
</body>
</html>
'''
    
    with open(output_file, "w") as f:
        f.write(html_content)
    
    print(f"✓ HTML visualization saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Create HTML trajectory visualization")
    parser.add_argument("--input_dir", type=str, default="./data_collection",
                        help="Directory with HDF5 data files")
    parser.add_argument("--output", type=str, default="trajectory_visualization.html",
                        help="Output HTML file")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    data_files = sorted(input_dir.glob("run_*.h5"))
    
    if not data_files:
        print(f"❌ No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(data_files)} data files")
    
    output_file = input_dir / args.output
    create_html_visualization(data_files, output_file)


if __name__ == "__main__":
    main()
