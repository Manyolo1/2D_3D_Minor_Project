// Simple Express backend to run single-image 3D reconstruction
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 4000;
const PYTHON_BIN = process.env.PYTHON_BIN || 'python';
const MODEL_DIR = process.env.MODEL_DIR || 'results_run1';
const EPOCH = process.env.EPOCH || '80';
const CATEGORY = process.env.CATEGORY || '0';
const RGB_DEPTH_MODEL = process.env.RGB_DEPTH_MODEL || 'midas-small';
const ROOT = path.join(__dirname, '..');
const OUTPUT_ROOT = path.join(__dirname, '..', 'web_output');

app.use(cors());
app.use(express.json());

// Multer setup for uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const dest = path.join(OUTPUT_ROOT, 'uploads');
        fs.mkdirSync(dest, { recursive: true });
        cb(null, dest);
    },
    filename: (req, file, cb) => {
        const id = uuidv4();
        const ext = path.extname(file.originalname) || '.png';
        cb(null, `${id}${ext}`);
    }
});

const upload = multer({ storage });

// Serve static outputs (meshes, point clouds, images)
app.use('/static', express.static(OUTPUT_ROOT, { fallthrough: true }));

// Friendly root handler so hitting "/" does not show "Cannot GET"
app.get('/', (_req, res) => {
    res.json({
        message: '3D reconstruction backend is running',
        health: '/api/health',
        reconstruct: 'POST /api/reconstruct (multipart field "image")'
    });
});

app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', modelDir: MODEL_DIR, epoch: EPOCH });
});

app.post('/api/reconstruct', upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }

    const inputPath = req.file.path;
    const jobId = path.basename(inputPath, path.extname(inputPath));
    const outputDir = path.join(OUTPUT_ROOT, jobId);

    fs.mkdirSync(outputDir, { recursive: true });

    // Build python command
    const scriptPath = path.join(ROOT, 'Main_Py', 'single_image_to_3d.py');
    const depthFromRGB = req.body.depthFromRGB === '1' || req.body.depthFromRGB === 'true';
    const rgbDepthModel = req.body.rgbDepthModel || RGB_DEPTH_MODEL;

    const args = [
        scriptPath,
        '--input', inputPath,
        '--model-dir', MODEL_DIR,
        '--epoch', EPOCH,
        '--category', req.body.category || CATEGORY,
        '--output-dir', outputDir,
    ];

    if (depthFromRGB) {
        args.push('--is-depth', '0', '--depth-from-rgb', '--rgb-depth-model', rgbDepthModel);
    }

    const py = spawn(PYTHON_BIN, args, { cwd: ROOT });

    let stdOut = '';
    let stdErr = '';

    py.stdout.on('data', (data) => { stdOut += data.toString(); });
    py.stderr.on('data', (data) => { stdErr += data.toString(); });

    py.on('close', (code) => {
        if (code !== 0) {
            console.error('Reconstruction failed:', stdErr);
            return res.status(500).json({ error: 'Reconstruction failed', stdErr, stdOut, code });
        }

        // Build URLs for front-end
        const meshRel = path.join(jobId, 'mesh_fused.ply');
        const cloudRel = path.join(jobId, 'pointcloud_fused.ply');
        const meshUrl = `/static/${meshRel.replace(/\\/g, '/')}`;
        const cloudUrl = `/static/${cloudRel.replace(/\\/g, '/')}`;

        res.json({
            jobId,
            meshUrl,
            cloudUrl,
            log: stdOut,
        });
    });
});

app.listen(PORT, () => {
    console.log(`Server listening on http://localhost:${PORT}`);
});
