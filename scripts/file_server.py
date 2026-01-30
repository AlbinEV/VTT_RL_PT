#!/usr/bin/env python3
"""
Server HTTP per convertire automaticamente H5/NPZ in JSON.
Avvia con: python scripts/file_server.py
"""
import _path_setup

import json
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import h5py
from pathlib import Path
import cgi


def convert_h5_to_dict(file_bytes):
    """Converte bytes H5 in dizionario."""
    with io.BytesIO(file_bytes) as buf:
        with h5py.File(buf, 'r') as f:
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
                raise ValueError(f"Campi non trovati. Disponibili: {list(f.keys())}")
            
            time_data = f[time_field][:]
            force_data = f[force_field][:]
            
            if len(force_data.shape) > 1:
                force_data = force_data[:, -1]
            
            return {
                'time': time_data.tolist(),
                'force': force_data.tolist()
            }


def convert_npz_to_dict(file_bytes):
    """Converte bytes NPZ in dizionario."""
    with io.BytesIO(file_bytes) as buf:
        data = np.load(buf)
        
        time_field = None
        force_field = None
        
        for field in ['timestamps', 'time', 'sim_time', 't']:
            if field in data:
                time_field = field
                break
        
        for field in ['contact_forces', 'force', 'fz', 'forces']:
            if field in data:
                force_field = field
                break
        
        if not time_field or not force_field:
            raise ValueError(f"Campi non trovati. Disponibili: {list(data.keys())}")
        
        time_data = data[time_field]
        force_data = data[force_field]
        
        if len(force_data.shape) > 1:
            force_data = force_data[:, -1]
        
        return {
            'time': time_data.tolist(),
            'force': force_data.tolist()
        }


class FileHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        if self.path == '/convert':
            try:
                content_type = self.headers['Content-Type']
                if 'multipart/form-data' not in content_type:
                    self.send_error(400, 'Expected multipart/form-data')
                    return
                
                # Parse form data
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST'}
                )
                
                if 'file' not in form:
                    self.send_error(400, 'No file uploaded')
                    return
                
                file_item = form['file']
                filename = file_item.filename
                file_bytes = file_item.file.read()
                
                # Converti in base all'estensione
                ext = Path(filename).suffix.lower()
                
                if ext == '.h5':
                    data = convert_h5_to_dict(file_bytes)
                elif ext == '.npz':
                    data = convert_npz_to_dict(file_bytes)
                elif ext == '.json':
                    data = json.loads(file_bytes.decode('utf-8'))
                else:
                    self.send_error(400, f'Formato non supportato: {ext}')
                    return
                
                # Invia risposta JSON
                response = json.dumps(data).encode('utf-8')
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', len(response))
                self.end_headers()
                self.wfile.write(response)
                
                print(f"✅ Convertito: {filename} ({len(data['time'])} punti)")
                
            except Exception as e:
                print(f"❌ Errore: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404, 'Endpoint non trovato')
    
    def log_message(self, format, *args):
        # Silenzia i log noiosi
        pass


def main():
    port = 8000
    server = HTTPServer(('localhost', port), FileHandler)
    print(f"🚀 Server avviato su http://localhost:{port}")
    print(f"📊 Apri force_comparator.html nel browser")
    print(f"   Ora puoi caricare direttamente file H5 e NPZ!")
    print(f"\n   Premi Ctrl+C per fermare il server\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Server fermato")


if __name__ == '__main__':
    main()
