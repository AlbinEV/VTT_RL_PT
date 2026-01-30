#!/usr/bin/env python3
"""
Script per convertire file H5 e NPZ in JSON per il visualizzatore HTML.

Usage:
    python scripts/export_to_json.py <input_file> [output_file]
"""
import _path_setup

import sys
import json
import numpy as np
import h5py
from pathlib import Path


def convert_h5_to_json(h5_path, output_path=None):
    """Converte un file H5 in JSON."""
    if output_path is None:
        output_path = h5_path.replace('.h5', '.json')
    
    with h5py.File(h5_path, 'r') as f:
        # Cerca campi tempo e forza
        time_field = None
        force_field = None
        
        for field in ['sim_time', 'time', 'timestamps', 't']:
            if field in f:
                time_field = field
                break
        
        for field in ['fz', 'force', 'contact_force', 'contact_forces']:
            if field in f:
                force_field = field
                break
        
        if not time_field or not force_field:
            raise ValueError(f"Campi tempo/forza non trovati. Disponibili: {list(f.keys())}")
        
        time_data = f[time_field][:]
        force_data = f[force_field][:]
        
        # Se forza è multidimensionale, prendi solo fz (indice 2 o ultimo)
        if len(force_data.shape) > 1:
            force_data = force_data[:, -1]  # Prendi ultima colonna
        
        data = {
            'name': Path(h5_path).stem,
            'source': 'H5',
            'time': time_data.tolist(),
            'force': force_data.tolist(),
            'metadata': {
                'duration': float(time_data[-1] - time_data[0]),
                'steps': len(time_data),
                'mean_force': float(np.mean(force_data)),
                'min_force': float(np.min(force_data)),
                'max_force': float(np.max(force_data))
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Convertito: {h5_path} -> {output_path}")
    print(f"   Durata: {data['metadata']['duration']:.2f}s, Steps: {data['metadata']['steps']}")
    print(f"   Forza: Mean={data['metadata']['mean_force']:.2f}N, Range=[{data['metadata']['min_force']:.2f}, {data['metadata']['max_force']:.2f}]N")
    
    return output_path


def convert_npz_to_json(npz_path, output_path=None):
    """Converte un file NPZ in JSON."""
    if output_path is None:
        output_path = npz_path.replace('.npz', '.json')
    
    data_npz = np.load(npz_path)
    
    # Cerca campi tempo e forza
    time_field = None
    force_field = None
    
    for field in ['timestamps', 'time', 'sim_time', 't']:
        if field in data_npz:
            time_field = field
            break
    
    for field in ['contact_forces', 'force', 'fz', 'forces']:
        if field in data_npz:
            force_field = field
            break
    
    if not time_field or not force_field:
        raise ValueError(f"Campi tempo/forza non trovati. Disponibili: {list(data_npz.keys())}")
    
    time_data = data_npz[time_field]
    force_data = data_npz[force_field]
    
    # Se forza è multidimensionale, prendi solo fz
    if len(force_data.shape) > 1:
        force_data = force_data[:, -1]
    
    data = {
        'name': Path(npz_path).stem,
        'source': 'NPZ',
        'time': time_data.tolist(),
        'force': force_data.tolist(),
        'metadata': {
            'duration': float(time_data[-1] - time_data[0]),
            'steps': len(time_data),
            'mean_force': float(np.mean(force_data)),
            'min_force': float(np.min(force_data)),
            'max_force': float(np.max(force_data))
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Convertito: {npz_path} -> {output_path}")
    print(f"   Durata: {data['metadata']['duration']:.2f}s, Steps: {data['metadata']['steps']}")
    print(f"   Forza: Mean={data['metadata']['mean_force']:.2f}N, Range=[{data['metadata']['min_force']:.2f}, {data['metadata']['max_force']:.2f}]N")
    
    return output_path


def convert_directory(directory, pattern='*.h5'):
    """Converte tutti i file in una directory."""
    directory = Path(directory)
    files = list(directory.glob(pattern))
    
    if not files:
        print(f"❌ Nessun file {pattern} trovato in {directory}")
        return
    
    print(f"\n🔄 Conversione di {len(files)} file da {directory}...\n")
    
    for file_path in files:
        try:
            if file_path.suffix == '.h5':
                convert_h5_to_json(str(file_path))
            elif file_path.suffix == '.npz':
                convert_npz_to_json(str(file_path))
        except Exception as e:
            print(f"❌ Errore con {file_path.name}: {e}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_to_json.py <input_file|directory> [output_file]")
        print("\nEsempi:")
        print("  python scripts/export_to_json.py data_fixed_mode/run_0000.h5")
        print("  python scripts/export_to_json.py data_fixed_mode/  # Converte tutti i .h5")
        print("  python scripts/export_to_json.py trajectories/episode_0.npz output.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Se è una directory, converti tutti i file
    if Path(input_path).is_dir():
        convert_directory(input_path, '*.h5')
        convert_directory(input_path, '*.npz')
        return
    
    # Altrimenti converti il singolo file
    ext = Path(input_path).suffix.lower()
    
    try:
        if ext == '.h5':
            convert_h5_to_json(input_path, output_path)
        elif ext == '.npz':
            convert_npz_to_json(input_path, output_path)
        else:
            print(f"❌ Formato non supportato: {ext}")
            print("   Supportati: .h5, .npz")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Errore durante la conversione: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
