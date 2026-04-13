const video = document.getElementById('webcam');
const canvas = document.getElementById('captureCanvas');
const context = canvas.getContext('2d');
const loader = document.getElementById('loader');
const predictionText = document.getElementById('prediction-text');
const aiStatus = document.getElementById('ai-status');
const aiLatency = document.getElementById('ai-latency');
const statusIndicator = document.getElementById('status-indicator');

// Wait precisely ensuring the camera dimensions have loaded before extracting frames
let isVideoReady = false;

// Setup the webcam stream!
async function setupWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 480 },
                facingMode: "user" // Selfie camera
            } 
        });
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
             canvas.width = video.videoWidth;
             canvas.height = video.videoHeight;
             video.play();
        }

        // Wait until video is absolutely playing correctly
        video.onplaying = () => {
             loader.classList.add('hidden');
             statusIndicator.classList.add('online');
             aiStatus.innerText = "Scanning Environment";
             isVideoReady = true;
             
             // Begin the infinite AI polling loop!
             pollAI();
        };
    } catch (e) {
        console.error("Webcam Error:", e);
        aiStatus.innerText = "Camera Denied/Offline";
        aiStatus.style.color = "#ff3333";
        loader.innerHTML = "<p style='color:#ff3333'>Camera Access Denied.<br/>Please allow camera permissions.</p>";
    }
}

async function pollAI() {
    if (!isVideoReady) return;
    
    // Draw current video frame strictly to the hidden canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to highly compressed JPEG base64
    // Quality 0.5 dramatically reduces the AJAX payload size, yielding incredible latency
    const base64Image = canvas.toDataURL('image/jpeg', 0.5);
    
    const startTime = Date.now();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Image })
        });
        
        const data = await response.json();
        const latency = Date.now() - startTime;
        
        aiLatency.innerText = latency + 'ms';
        
        if (data.success && data.prediction !== null) {
            aiStatus.innerText = "Target Locked!";
            aiStatus.style.color = "var(--accent)";
            
            // Only bump if the prediction actually changed!
            if (predictionText.innerText !== data.prediction) {
                predictionText.innerText = data.prediction;
                // Add a cool css pop animation
                predictionText.style.transform = "scale(1.2)";
                setTimeout(() => { predictionText.style.transform = "scale(1)"; }, 150);
            }
        } else {
            // "No hand detected" or MediaPipe inherently failed that frame
            aiStatus.innerText = "Scanning...";
            aiStatus.style.color = "var(--text-sec)";
            predictionText.innerText = "-";
        }
    } catch (error) {
        console.error("Inference Error:", error);
        aiStatus.innerText = "Server Disconnected";
        aiStatus.style.color = "#ff3333";
    }
    
    // Recursive polling every ~150 milliseconds! 
    // This allows Python to easily keep up without overwhelming the Flask server.
    setTimeout(pollAI, 150);
}

// Kickstart the magic!
setupWebcam();
