#!/usr/bin/env python3
"""
Create interactive HTML visualization of Fixed Mode trajectory data.
Shows 3D trajectory, force profiles, impedance parameters, and waypoints.
"""
import _path_setup

import argparse
import os
import glob
import h5py
import numpy as np
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Fixed Mode Trajectory Visualization</title>
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
        }}
        .header h1 {{
            font-size: 28px;
            color: #00d9ff;
            margin-bottom: 5px;
        }}
        .header .subtitle {{
            color: #888;
            font-size: 14px;
        }}
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 15px 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-card .label {{
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #00d9ff;
            margin-top: 5px;
        }}
        .stat-card .unit {{
            font-size: 14px;
            color: #666;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .plot-card {{
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .plot-card.full-width {{
            grid-column: span 2;
        }}
        .plot-card h3 {{
            color: #00d9ff;
            font-size: 16px;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .waypoint-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
            padding: 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }}
        .waypoint-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 5px 12px;
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            font-size: 13px;
        }}
        .waypoint-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        .phase-indicator {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            margin-left: 5px;
        }}
        .phase-approach {{ background: #e74c3c; }}
        .phase-contact {{ background: #2ecc71; }}
        .phase-lateral {{ background: #3498db; }}
        .phase-rise {{ background: #9b59b6; }}
        @media (max-width: 1200px) {{
            .plot-grid {{ grid-template-columns: 1fr; }}
            .plot-card.full-width {{ grid-column: span 1; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Fixed Mode Trajectory Analysis</h1>
        <div class="subtitle">VTT_RL - Variable Impedance Control with Discrete Waypoints</div>
    </div>
    
    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Total Steps</div>
                <div class="value">{total_steps}</div>
            </div>
            <div class="stat-card">
                <div class="label">Duration</div>
                <div class="value">{duration:.1f} <span class="unit">s</span></div>
            </div>
            <div class="stat-card">
                <div class="label">Waypoints</div>
                <div class="value">{num_waypoints}</div>
            </div>
            <div class="stat-card">
                <div class="label">Contact Steps</div>
                <div class="value">{contact_steps} <span class="unit">({contact_pct:.0f}%)</span></div>
            </div>
            <div class="stat-card">
                <div class="label">Mean |Fz|</div>
                <div class="value">{mean_fz:.2f} <span class="unit">N</span></div>
            </div>
            <div class="stat-card">
                <div class="label">Total Reward</div>
                <div class="value">{total_reward:.1f}</div>
            </div>
        </div>
        
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
                <h3>🔧 Stiffness Kp_z</h3>
                <div id="plot-kp"></div>
            </div>
            
            <div class="plot-card">
                <h3>🌊 Damping ζ_z</h3>
                <div id="plot-zeta"></div>
            </div>
            
            <div class="plot-card full-width">
                <h3>🎯 Waypoint Index & Phase</h3>
                <div id="plot-waypoint"></div>
            </div>
            
            <div class="plot-card full-width">
                <h3>💰 Cumulative Reward</h3>
                <div id="plot-reward"></div>
            </div>
        </div>
    </div>
    
    <script>
        const data = {json_data};
        
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
            font: {{ color: '#e8e8e8', size: 11 }},
            margin: {{ l: 50, r: 30, t: 30, b: 40 }},
            xaxis: {{
                gridcolor: colors.grid,
                zerolinecolor: colors.grid,
                title: {{ text: 'Step', standoff: 10 }}
            }},
            yaxis: {{
                gridcolor: colors.grid,
                zerolinecolor: colors.grid
            }},
            showlegend: true,
            legend: {{ x: 1, y: 1, xanchor: 'right', bgcolor: 'rgba(0,0,0,0.3)' }}
        }};
        
        // 3D Trajectory Plot
        const trace3d = {{
            type: 'scatter3d',
            mode: 'lines+markers',
            x: data.ee_pos_x,
            y: data.ee_pos_y,
            z: data.ee_pos_z,
            marker: {{
                size: 3,
                color: data.wpt_idx,
                colorscale: waypointColors.map((c, i) => [i/7, c]).concat([[1, waypointColors[7]]]),
                cmin: 0,
                cmax: 7
            }},
            line: {{ width: 2, color: colors.primary }},
            name: 'Trajectory',
            hovertemplate: 'X: %{{x:.4f}}<br>Y: %{{y:.4f}}<br>Z: %{{z:.4f}}<br>WPT: %{{marker.color}}<extra></extra>'
        }};
        
        // Waypoint markers
        const waypointMarkers = {{
            type: 'scatter3d',
            mode: 'markers+text',
            x: data.waypoint_x,
            y: data.waypoint_y,
            z: data.waypoint_z,
            marker: {{
                size: 10,
                color: waypointColors.slice(0, data.waypoint_x.length),
                symbol: 'diamond'
            }},
            text: data.waypoint_labels,
            textposition: 'top center',
            textfont: {{ size: 10, color: '#fff' }},
            name: 'Waypoints',
            hovertemplate: '%{{text}}<br>(%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}})<extra></extra>'
        }};
        
        Plotly.newPlot('plot-3d', [trace3d, waypointMarkers], {{
            ...defaultLayout,
            height: 500,
            scene: {{
                xaxis: {{ title: 'X (m)', gridcolor: colors.grid, backgroundcolor: colors.plot }},
                yaxis: {{ title: 'Y (m)', gridcolor: colors.grid, backgroundcolor: colors.plot }},
                zaxis: {{ title: 'Z (m)', gridcolor: colors.grid, backgroundcolor: colors.plot }},
                camera: {{ eye: {{ x: 1.5, y: 1.5, z: 1.2 }} }},
                aspectmode: 'data'
            }}
        }});
        
        // Position Plot
        Plotly.newPlot('plot-pos', [
            {{ y: data.ee_pos_x, name: 'X', line: {{ color: '#e74c3c' }} }},
            {{ y: data.ee_pos_y, name: 'Y', line: {{ color: '#2ecc71' }} }},
            {{ y: data.ee_pos_z, name: 'Z', line: {{ color: '#3498db' }} }}
        ], {{
            ...defaultLayout,
            height: 300,
            yaxis: {{ ...defaultLayout.yaxis, title: 'Position (m)' }}
        }});
        
        // Force Plot
        Plotly.newPlot('plot-force', [
            {{ y: data.fz, name: 'Fz', line: {{ color: colors.secondary }}, fill: 'tozeroy' }},
            {{ y: data.fz.map(() => -2), name: 'Target (-2N)', line: {{ color: '#fff', dash: 'dash' }} }}
        ], {{
            ...defaultLayout,
            height: 300,
            yaxis: {{ ...defaultLayout.yaxis, title: 'Force (N)' }}
        }});
        
        // Kp Plot
        Plotly.newPlot('plot-kp', [
            {{ y: data.kp_z, name: 'Kp_z', line: {{ color: colors.tertiary }} }}
        ], {{
            ...defaultLayout,
            height: 300,
            yaxis: {{ ...defaultLayout.yaxis, title: 'Stiffness (N/m)' }}
        }});
        
        // Zeta Plot
        Plotly.newPlot('plot-zeta', [
            {{ y: data.zeta_z, name: 'ζ_z', line: {{ color: colors.quaternary }} }}
        ], {{
            ...defaultLayout,
            height: 300,
            yaxis: {{ ...defaultLayout.yaxis, title: 'Damping Ratio' }}
        }});
        
        // Waypoint Plot
        Plotly.newPlot('plot-waypoint', [
            {{ y: data.wpt_idx, name: 'Waypoint', line: {{ color: colors.primary }}, fill: 'tozeroy' }},
            {{ y: data.phase, name: 'Phase', line: {{ color: colors.secondary, dash: 'dot' }}, yaxis: 'y2' }}
        ], {{
            ...defaultLayout,
            height: 250,
            yaxis: {{ ...defaultLayout.yaxis, title: 'Waypoint Index', range: [-0.5, 8] }},
            yaxis2: {{ title: 'Phase', overlaying: 'y', side: 'right', gridcolor: 'rgba(0,0,0,0)' }}
        }});
        
        // Reward Plot
        const cumReward = data.reward.reduce((acc, r, i) => {{
            acc.push((acc[i-1] || 0) + r);
            return acc;
        }}, []);
        
        Plotly.newPlot('plot-reward', [
            {{ y: cumReward, name: 'Cumulative Reward', line: {{ color: '#f1c40f' }}, fill: 'tozeroy' }}
        ], {{
            ...defaultLayout,
            height: 250,
            yaxis: {{ ...defaultLayout.yaxis, title: 'Cumulative Reward' }}
        }});
    </script>
</body>
</html>
"""


def load_h5_data(filepath: str) -> dict:
    """Load trajectory data from H5 file."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            data[key] = f[key][:]
    return data


def merge_runs(data_dir: str) -> dict:
    """Merge multiple run files into single dataset."""
    h5_files = sorted(glob.glob(os.path.join(data_dir, "run_*.h5")))
    
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {data_dir}")
    
    print(f"Found {len(h5_files)} run files")
    
    # Load and concatenate
    all_data = {}
    for filepath in h5_files:
        run_data = load_h5_data(filepath)
        for key, arr in run_data.items():
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(arr)
    
    # Concatenate arrays
    merged = {}
    for key, arrays in all_data.items():
        merged[key] = np.concatenate(arrays, axis=0)
    
    return merged


def create_visualization(data_dir: str, output_path: str):
    """Create interactive HTML visualization."""
    
    print(f"Loading data from: {data_dir}")
    data = merge_runs(data_dir)
    
    total_steps = len(data['frame'])
    print(f"Total steps: {total_steps}")
    
    # Calculate statistics
    duration = data['sim_time'][-1] - data['sim_time'][0] if 'sim_time' in data else total_steps * 0.1
    contact_mask = np.abs(data['fz']) > 0.5
    contact_steps = np.sum(contact_mask)
    contact_pct = 100 * contact_steps / total_steps
    mean_fz = np.mean(np.abs(data['fz'][contact_mask])) if contact_steps > 0 else 0
    total_reward = np.sum(data['reward']) if 'reward' in data else 0
    
    # Get unique waypoints
    unique_wpts = np.unique(data['wpt_idx'])
    num_waypoints = len(unique_wpts)
    
    # Define waypoint positions (from Fixed Mode config)
    waypoint_positions = [
        [0.55, 0.0, 0.55],   # HOME
        [0.55, 0.0, 0.325],  # INTERMEDIATE
        [0.55, 0.0, 0.1],    # APPROACH
        [0.55, 0.0, 0.0],    # CONTACT
        [0.55, -0.2, 0.0],   # LATERAL_0
        [0.55, 0.0, 0.0],    # LATERAL_1
        [0.55, 0.2, 0.0],    # LATERAL_2
        [0.55, 0.0, 0.15],   # RISE
    ]
    waypoint_labels = ['HOME', 'INTERM', 'APPROACH', 'CONTACT', 'LAT_0', 'LAT_1', 'LAT_2', 'RISE']
    
    # Prepare JSON data
    json_data = {
        'ee_pos_x': data['ee_pos'][:, 0].tolist(),
        'ee_pos_y': data['ee_pos'][:, 1].tolist(),
        'ee_pos_z': data['ee_pos'][:, 2].tolist(),
        'fz': data['fz'].tolist(),
        'kp_z': data['kp'][:, 2].tolist() if data['kp'].shape[1] > 2 else data['kp'][:, 0].tolist(),
        'zeta_z': data['zeta'][:, 2].tolist() if data['zeta'].shape[1] > 2 else data['zeta'][:, 0].tolist(),
        'wpt_idx': data['wpt_idx'].tolist(),
        'phase': data['phase'].tolist() if 'phase' in data else [0] * total_steps,
        'reward': data['reward'].tolist() if 'reward' in data else [0] * total_steps,
        'waypoint_x': [wp[0] for wp in waypoint_positions],
        'waypoint_y': [wp[1] for wp in waypoint_positions],
        'waypoint_z': [wp[2] for wp in waypoint_positions],
        'waypoint_labels': waypoint_labels,
    }
    
    import json
    json_str = json.dumps(json_data)
    
    # Generate HTML
    html = HTML_TEMPLATE.format(
        total_steps=total_steps,
        duration=duration,
        num_waypoints=num_waypoints,
        contact_steps=contact_steps,
        contact_pct=contact_pct,
        mean_fz=mean_fz,
        total_reward=total_reward,
        json_data=json_str
    )
    
    # Write output
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"\n✅ Visualization saved to: {output_path}")
    print(f"   Open in browser to view interactive plots")


def main():
    parser = argparse.ArgumentParser(description="Create trajectory visualization HTML")
    parser.add_argument("--input_dir", type=str, default="./data_fixed_mode",
                        help="Directory containing H5 run files")
    parser.add_argument("--output", type=str, default="trajectory_visualization.html",
                        help="Output HTML file path")
    args = parser.parse_args()
    
    create_visualization(args.input_dir, args.output)


if __name__ == "__main__":
    main()
