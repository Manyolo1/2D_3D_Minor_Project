import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three-stdlib';
import { PLYLoader } from 'three-stdlib';

export default function PLYViewer({ url }) {
    const mountRef = useRef(null);

    useEffect(() => {
        if (!url) return;

        const mount = mountRef.current;
        const width = mount.clientWidth;
        const height = mount.clientHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color('#111');

        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.set(1, 1, 2);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        mount.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
        scene.add(hemiLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(3, 5, 2);
        scene.add(dirLight);

        const loader = new PLYLoader();
        loader.load(url, (geometry) => {
            geometry.computeVertexNormals();
            const material = new THREE.MeshStandardMaterial({ color: 0x4fc3f7, flatShading: false });
            const mesh = new THREE.Mesh(geometry, material);
            geometry.center();
            scene.add(mesh);

            // Frame the object
            const box = new THREE.Box3().setFromObject(mesh);
            const size = box.getSize(new THREE.Vector3()).length();
            const center = box.getCenter(new THREE.Vector3());
            controls.target.copy(center);
            camera.position.copy(center.clone().add(new THREE.Vector3(size, size, size)));
            camera.lookAt(center);
            controls.update();
        });

        const handleResize = () => {
            const w = mount.clientWidth;
            const h = mount.clientHeight;
            renderer.setSize(w, h);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        };
        window.addEventListener('resize', handleResize);

        const animate = () => {
            controls.update();
            renderer.render(scene, camera);
            requestAnimationFrame(animate);
        };
        animate();

        return () => {
            window.removeEventListener('resize', handleResize);
            renderer.dispose();
            mount.removeChild(renderer.domElement);
        };
    }, [url]);

    return <div className="ply-view" ref={mountRef} />;
}
