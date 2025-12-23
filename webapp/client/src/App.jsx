import React, { useState } from 'react';
import axios from 'axios';
import PLYViewer from './PLYViewer';

const API_BASE = '';

export default function App() {
    const [file, setFile] = useState(null);
    const [category, setCategory] = useState('0');
    const [status, setStatus] = useState('Idle');
    const [meshUrl, setMeshUrl] = useState('');
    const [cloudUrl, setCloudUrl] = useState('');
    const [log, setLog] = useState('');
    const [depthFromRGB, setDepthFromRGB] = useState(true);
    const [rgbDepthModel, setRgbDepthModel] = useState('midas-small');

    const onSubmit = async (e) => {
        e.preventDefault();
        if (!file) {
            alert('Please choose an image');
            return;
        }
        setStatus('Uploading...');
        setMeshUrl('');
        setCloudUrl('');
        setLog('');

        try {
            const form = new FormData();
            form.append('image', file);
            form.append('category', category);
            form.append('depthFromRGB', depthFromRGB ? '1' : '0');
            form.append('rgbDepthModel', rgbDepthModel);

            setStatus('Reconstructing (this may take ~30-90s)...');
            const res = await axios.post(`${API_BASE}/api/reconstruct`, form, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            setStatus('Done');
            setMeshUrl(res.data.meshUrl);
            setCloudUrl(res.data.cloudUrl);
            setLog(res.data.log || '');
        } catch (err) {
            console.error(err);
            setStatus('Error');
            const msg = err.response?.data?.error || err.message;
            setLog(msg + '\n' + (err.response?.data?.stdErr || ''));
        }
    };

    return (
        <div className="page">
            <header>
                <h1>Single Image → 3D Reconstruction</h1>
                <p>Upload one 2D image (depth or RGB) and get a 3D mesh. For room photos, keep "Estimate depth from RGB" on.</p>
            </header>

            <form className="card" onSubmit={onSubmit}>
                <label className="field">
                    <span>Choose 2D image</span>
                    <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files[0])} />
                </label>

                <label className="field">
                    <span>Category (0=cube, 1=sphere)</span>
                    <select value={category} onChange={(e) => setCategory(e.target.value)}>
                        <option value="0">Cube</option>
                        <option value="1">Sphere</option>
                    </select>
                </label>

                <label className="field checkbox">
                    <input type="checkbox" checked={depthFromRGB} onChange={(e) => setDepthFromRGB(e.target.checked)} />
                    <span>Estimate depth from RGB (enable for normal photos)</span>
                </label>

                <label className="field">
                    <span>Depth model</span>
                    <select value={rgbDepthModel} onChange={(e) => setRgbDepthModel(e.target.value)} disabled={!depthFromRGB}>
                        <option value="midas-small">MiDaS small (fast)</option>
                        <option value="midas-large">MiDaS large (better quality)</option>
                    </select>
                </label>

                <button type="submit">Reconstruct</button>
                <div className="status">Status: {status}</div>
            </form>

            {meshUrl && (
                <div className="card viewer">
                    <div className="viewer-header">
                        <h2>3D Mesh</h2>
                        <div className="links">
                            <a href={meshUrl} target="_blank" rel="noreferrer">Download mesh</a>
                            <a href={cloudUrl} target="_blank" rel="noreferrer">Download point cloud</a>
                        </div>
                    </div>
                    <PLYViewer url={meshUrl} />
                </div>
            )}

            {log && (
                <div className="card log">
                    <h3>Server log</h3>
                    <pre>{log}</pre>
                </div>
            )}
        </div>
    );
}
