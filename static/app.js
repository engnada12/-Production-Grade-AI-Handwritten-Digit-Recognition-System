/**
 * Handwritten Digit Recognizer - Client Application Logic
 * High-Performance Multi-Modal Interaction (Canvas, Camera, Upload)
 */

document.addEventListener('DOMContentLoaded', () => {
    // State management
    const state = {
        activeTab: 'drawingTab',
        isDrawing: false,
        lastX: 0,
        lastY: 0,
        canvasHasDrawing: false,
        cameraStream: null,
        currentFacingMode: 'environment',
        lastPredictionResult: null,
        lastInputImageBase64: null
    };

    // DOM Elements
    const elements = {
        // Tabs
        tabBtns: document.querySelectorAll('.tab-btn'),
        tabPanes: document.querySelectorAll('.tab-pane'),
        
        // Canvas
        canvas: document.getElementById('digitCanvas'),
        canvasHint: document.getElementById('canvasHint'),
        brushSize: document.getElementById('brushSize'),
        clearCanvasBtn: document.getElementById('clearCanvasBtn'),
        recognizeCanvasBtn: document.getElementById('recognizeCanvasBtn'),
        
        // Camera
        cameraFeed: document.getElementById('cameraFeed'),
        cameraFallback: document.getElementById('cameraFallback'),
        startCameraBtn: document.getElementById('startCameraBtn'),
        switchCameraBtn: document.getElementById('switchCameraBtn'),
        captureBtn: document.getElementById('captureBtn'),
        cameraToolbar: document.getElementById('cameraToolbar'),
        cameraCaptureCanvas: document.getElementById('cameraCaptureCanvas'),
        
        // Upload
        dropZone: document.getElementById('dropZone'),
        fileInput: document.getElementById('fileInput'),
        dropzonePrompt: document.getElementById('dropzonePrompt'),
        uploadPreviewWrapper: document.getElementById('uploadPreviewWrapper'),
        uploadPreviewImg: document.getElementById('uploadPreviewImg'),
        removeUploadBtn: document.getElementById('removeUploadBtn'),
        uploadActions: document.getElementById('uploadActions'),
        recognizeUploadBtn: document.getElementById('recognizeUploadBtn'),
        
        // States & Results
        stateIdle: document.getElementById('stateIdle'),
        stateLoading: document.getElementById('stateLoading'),
        stateResult: document.getElementById('stateResult'),
        predictedDigitDisplay: document.getElementById('predictedDigitDisplay'),
        confidenceTag: document.getElementById('confidenceTag'),
        preprocessedPreviewImg: document.getElementById('preprocessedPreviewImg'),
        metricTotalTime: document.getElementById('metricTotalTime'),
        metricInferenceTime: document.getElementById('metricInferenceTime'),
        metricModelType: document.getElementById('metricModelType'),
        probabilitiesGrid: document.getElementById('probabilitiesGrid'),
        resetResultBtn: document.getElementById('resetResultBtn'),
        
        // Feedback
        feedbackCorrectBtn: document.getElementById('feedbackCorrectBtn'),
        feedbackWrongBtn: document.getElementById('feedbackWrongBtn'),
        correctionPanel: document.getElementById('correctionPanel'),
        feedbackMsg: document.getElementById('feedbackMsg'),
        
        // Header & Modals
        activeModelName: document.getElementById('activeModelName'),
        infoModalBtn: document.getElementById('infoModalBtn'),
        infoModal: document.getElementById('infoModal'),
        closeModalBtn: document.getElementById('closeModalBtn')
    };

    // Canvas Context Setup
    const ctx = elements.canvas.getContext('2d');
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = parseInt(elements.brushSize.value, 10);

    // ==========================================
    // 1. Tab Switching & Lifecycle
    // ==========================================
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            if (state.activeTab === targetTab) return;

            // Update UI buttons
            elements.tabBtns.forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');

            // Switch panes
            elements.tabPanes.forEach(pane => pane.classList.remove('active'));
            document.getElementById(targetTab).classList.add('active');
            state.activeTab = targetTab;

            // Handle Camera stop if switching away from camera
            if (targetTab !== 'cameraTab' && state.cameraStream) {
                stopCamera();
            }
        });
    });

    // ==========================================
    // 2. Interactive Drawing Canvas
    // ==========================================
    function getCanvasCoordinates(e) {
        const rect = elements.canvas.getBoundingClientRect();
        const scaleX = elements.canvas.width / rect.width;
        const scaleY = elements.canvas.height / rect.height;

        let clientX = e.clientX;
        let clientY = e.clientY;

        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function startDrawing(e) {
        e.preventDefault();
        state.isDrawing = true;
        const coords = getCanvasCoordinates(e);
        state.lastX = coords.x;
        state.lastY = coords.y;

        // Draw initial dot
        ctx.beginPath();
        ctx.arc(state.lastX, state.lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
        ctx.fillStyle = '#FFFFFF';
        ctx.fill();

        if (elements.canvasHint) {
            elements.canvasHint.style.opacity = '0';
        }
        state.canvasHasDrawing = true;
    }

    function draw(e) {
        if (!state.isDrawing) return;
        e.preventDefault();

        const coords = getCanvasCoordinates(e);
        ctx.beginPath();
        ctx.moveTo(state.lastX, state.lastY);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();

        state.lastX = coords.x;
        state.lastY = coords.y;
    }

    function stopDrawing() {
        state.isDrawing = false;
    }

    // Mouse Listeners
    elements.canvas.addEventListener('mousedown', startDrawing);
    elements.canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);

    // Touch Listeners (Mobile & Tablet)
    elements.canvas.addEventListener('touchstart', startDrawing, { passive: false });
    elements.canvas.addEventListener('touchmove', draw, { passive: false });
    window.addEventListener('touchend', stopDrawing);
    window.addEventListener('touchcancel', stopDrawing);

    // Brush Size Slider
    elements.brushSize.addEventListener('input', (e) => {
        ctx.lineWidth = parseInt(e.target.value, 10);
    });

    // Clear Canvas
    function clearCanvas() {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, elements.canvas.width, elements.canvas.height);
        state.canvasHasDrawing = false;
        if (elements.canvasHint) {
            elements.canvasHint.style.opacity = '1';
        }
    }
    elements.clearCanvasBtn.addEventListener('click', clearCanvas);
    // Initialize black canvas
    clearCanvas();

    // Recognize Canvas Trigger
    elements.recognizeCanvasBtn.addEventListener('click', () => {
        if (!state.canvasHasDrawing) {
            alert('يرجى رسم رقم أولاً على اللوحة قبل التعرّف عليه.');
            return;
        }
        const base64Data = elements.canvas.toDataURL('image/png');
        sendPredictionRequest(base64Data);
    });

    // ==========================================
    // 3. Live WebRTC Camera Studio
    // ==========================================
    async function startCamera() {
        try {
            if (state.cameraStream) {
                stopCamera();
            }

            const constraints = {
                video: {
                    facingMode: state.currentFacingMode,
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            state.cameraStream = stream;
            elements.cameraFeed.srcObject = stream;
            elements.cameraFallback.style.display = 'none';
            elements.cameraToolbar.style.display = 'flex';
        } catch (err) {
            console.error('Camera access error:', err);
            alert('تعذر تشغيل الكاميرا. يرجى التأكد من منح الإذن لاستخدام الكاميرا في المتصفح.');
        }
    }

    function stopCamera() {
        if (state.cameraStream) {
            state.cameraStream.getTracks().forEach(track => track.stop());
            state.cameraStream = null;
            elements.cameraFeed.srcObject = null;
            elements.cameraFallback.style.display = 'flex';
            elements.cameraToolbar.style.display = 'none';
        }
    }

    elements.startCameraBtn.addEventListener('click', startCamera);

    elements.switchCameraBtn.addEventListener('click', () => {
        state.currentFacingMode = state.currentFacingMode === 'user' ? 'environment' : 'user';
        startCamera();
    });

    // Capture & Crop Target Area
    elements.captureBtn.addEventListener('click', () => {
        if (!state.cameraStream) return;

        const video = elements.cameraFeed;
        const captureCanvas = elements.cameraCaptureCanvas;
        captureCanvas.width = video.videoWidth || 640;
        captureCanvas.height = video.videoHeight || 480;

        const capCtx = captureCanvas.getContext('2d');
        capCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

        // Crop the central square guide area
        const cropSize = Math.min(captureCanvas.width, captureCanvas.height) * 0.65;
        const startX = (captureCanvas.width - cropSize) / 2;
        const startY = (captureCanvas.height - cropSize) / 2;

        const croppedCanvas = document.createElement('canvas');
        croppedCanvas.width = 280;
        croppedCanvas.height = 280;
        const croppedCtx = croppedCanvas.getContext('2d');
        croppedCtx.drawImage(captureCanvas, startX, startY, cropSize, cropSize, 0, 0, 280, 280);

        const base64Data = croppedCanvas.toDataURL('image/png');
        sendPredictionRequest(base64Data);
    });

    // ==========================================
    // 4. Upload & Drag-and-Drop Studio
    // ==========================================
    function handleSelectedFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            alert('يرجى اختيار ملف صورة صالح (PNG, JPEG, WebP).');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const base64Data = e.target.result;
            elements.uploadPreviewImg.src = base64Data;
            elements.dropzonePrompt.style.display = 'none';
            elements.uploadPreviewWrapper.style.display = 'block';
            elements.uploadActions.style.display = 'block';
            state.lastInputImageBase64 = base64Data;
        };
        reader.readAsDataURL(file);
    }

    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(evt => {
        elements.dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            elements.dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        elements.dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            elements.dropZone.classList.remove('dragover');
        });
    });

    elements.dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    // Clipboard Paste Listener (Ctrl+V)
    window.addEventListener('paste', (e) => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let item of items) {
            if (item.type.indexOf('image') !== -1) {
                const blob = item.getAsFile();
                // Switch to upload tab and load image
                document.getElementById('tabUploadBtn').click();
                handleSelectedFile(blob);
                break;
            }
        }
    });

    // Remove Upload
    elements.removeUploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        elements.uploadPreviewImg.src = '';
        elements.dropzonePrompt.style.display = 'block';
        elements.uploadPreviewWrapper.style.display = 'none';
        elements.uploadActions.style.display = 'none';
        elements.fileInput.value = '';
        state.lastInputImageBase64 = null;
    });

    elements.recognizeUploadBtn.addEventListener('click', () => {
        if (!state.lastInputImageBase64) return;
        sendPredictionRequest(state.lastInputImageBase64);
    });

    // ==========================================
    // 5. REST API Client & Prediction Pipeline
    // ==========================================
    async function sendPredictionRequest(base64Image) {
        state.lastInputImageBase64 = base64Image;

        // Switch Right Card to Loading state
        showState('loading');

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: base64Image })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.message || 'فشل الاستجابة من خادم الاستدلال.');
            }

            const data = await response.json();
            state.lastPredictionResult = data;
            renderPredictionResults(data);
            showState('result');

        } catch (error) {
            console.error('Prediction API Error:', error);
            alert(`حدث خطأ أثناء الاتصال بالخادم: ${error.message}`);
            showState('idle');
        }
    }

    function showState(stateName) {
        elements.stateIdle.classList.remove('active');
        elements.stateLoading.classList.remove('active');
        elements.stateResult.classList.remove('active');

        if (stateName === 'idle') elements.stateIdle.classList.add('active');
        if (stateName === 'loading') elements.stateLoading.classList.add('active');
        if (stateName === 'result') elements.stateResult.classList.add('active');
    }

    function renderPredictionResults(data) {
        // Hero predicted digit
        elements.predictedDigitDisplay.textContent = data.prediction;
        
        // Confidence
        const confidencePct = Math.round(data.confidence * 100);
        elements.confidenceTag.textContent = `ثقة التنبؤ: ${confidencePct}%`;
        
        if (confidencePct >= 90) {
            elements.confidenceTag.style.background = 'rgba(16, 185, 129, 0.2)';
            elements.confidenceTag.style.color = '#6ee7b7';
            elements.confidenceTag.style.borderColor = 'rgba(16, 185, 129, 0.4)';
        } else if (confidencePct >= 70) {
            elements.confidenceTag.style.background = 'rgba(245, 158, 11, 0.2)';
            elements.confidenceTag.style.color = '#fde68a';
            elements.confidenceTag.style.borderColor = 'rgba(245, 158, 11, 0.4)';
        } else {
            elements.confidenceTag.style.background = 'rgba(244, 63, 94, 0.2)';
            elements.confidenceTag.style.color = '#fda4af';
            elements.confidenceTag.style.borderColor = 'rgba(244, 63, 94, 0.4)';
        }

        // Preprocessed 28x28 preview image
        if (data.preprocessed_preview) {
            elements.preprocessedPreviewImg.src = data.preprocessed_preview;
        }

        // Latency Metrics
        const totalMs = data.timing.total_latency_ms;
        const inferMs = data.timing.inference_ms;
        elements.metricTotalTime.textContent = `${totalMs} ms`;
        elements.metricInferenceTime.textContent = `${inferMs} ms`;
        elements.metricModelType.textContent = data.model_type || 'CNN';

        // Render Probabilities Bar Chart (Digits 1 to 9)
        elements.probabilitiesGrid.innerHTML = '';
        const probs = data.probabilities;

        for (let digit = 1; digit <= 9; digit++) {
            const probVal = probs[digit] !== undefined ? probs[digit] : 0;
            const pct = Math.round(probVal * 100);
            const isWinner = digit === data.prediction;

            const row = document.createElement('div');
            row.className = `prob-row ${isWinner ? 'active' : ''}`;
            row.innerHTML = `
                <span class="prob-digit">${digit}</span>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width: 0%;"></div>
                </div>
                <span class="prob-pct">${pct}%</span>
            `;
            elements.probabilitiesGrid.appendChild(row);

            // Animate bar width smoothly
            setTimeout(() => {
                const fillBar = row.querySelector('.prob-bar-fill');
                if (fillBar) fillBar.style.width = `${pct}%`;
            }, 50);
        }

        // Reset feedback state
        elements.correctionPanel.style.display = 'none';
        elements.feedbackMsg.style.display = 'none';
        elements.feedbackMsg.textContent = '';
    }

    elements.resetResultBtn.addEventListener('click', () => {
        showState('idle');
        if (state.activeTab === 'drawingTab') {
            clearCanvas();
        }
    });

    // ==========================================
    // 6. Active Learning / Feedback Loop
    // ==========================================
    async function submitFeedback(actualDigit) {
        if (!state.lastInputImageBase64 || !state.lastPredictionResult) return;

        try {
            const res = await fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: state.lastInputImageBase64,
                    predicted_digit: state.lastPredictionResult.prediction,
                    actual_digit: actualDigit,
                    is_correct: actualDigit === state.lastPredictionResult.prediction
                })
            });

            if (res.ok) {
                elements.correctionPanel.style.display = 'none';
                elements.feedbackMsg.style.display = 'block';
                elements.feedbackMsg.textContent = '✓ شكراً لمساهمتك! تم حفظ العينة في مسار التعلم النشط.';
            }
        } catch (err) {
            console.error('Feedback submission error:', err);
        }
    }

    elements.feedbackCorrectBtn.addEventListener('click', () => {
        if (state.lastPredictionResult) {
            submitFeedback(state.lastPredictionResult.prediction);
        }
    });

    elements.feedbackWrongBtn.addEventListener('click', () => {
        elements.correctionPanel.style.display = 'flex';
    });

    document.querySelectorAll('.btn-pick').forEach(btn => {
        btn.addEventListener('click', () => {
            const actualDigit = parseInt(btn.getAttribute('data-digit'), 10);
            submitFeedback(actualDigit);
        });
    });

    // ==========================================
    // 7. System Health & Model Status Polling
    // ==========================================
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            if (res.ok) {
                const data = await res.json();
                elements.activeModelName.textContent = `${data.active_model} (متصل)`;
            } else {
                elements.activeModelName.textContent = 'غير متصل';
            }
        } catch (err) {
            elements.activeModelName.textContent = 'غير متصل';
        }
    }
    checkHealth();

    // ==========================================
    // 8. Technical Architecture Modal
    // ==========================================
    elements.infoModalBtn.addEventListener('click', () => {
        elements.infoModal.classList.add('active');
    });

    elements.closeModalBtn.addEventListener('click', () => {
        elements.infoModal.classList.remove('active');
    });

    elements.infoModal.addEventListener('click', (e) => {
        if (e.target === elements.infoModal) {
            elements.infoModal.classList.remove('active');
        }
    });
});
