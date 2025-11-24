import { useState, useEffect, useRef } from 'react';
import type { ChangeEvent, KeyboardEvent } from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import '@tensorflow/tfjs';

// --- Types ---
interface ChatMessage {
  role: 'user' | 'ai';
  text: string;
}
interface ImageData {
  base64: string | null;
  mimeType: string | null;
}
interface Prediction {
  className: string;
  probability: number;
}

// --- Icons ---
const IconUpload = () => (
  <svg className="w-12 h-12 mb-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
);
const IconCamera = () => (
  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
);
const IconRobot = () => (
  <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path></svg>
);
const IconSend = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>
);

function App() {
  // --- State ---
  const [model, setModel] = useState<mobilenet.MobileNet | null>(null);
  const [loadingModel, setLoadingModel] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<'upload' | 'webcam'>('upload');
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isWebcamActive, setIsWebcamActive] = useState<boolean>(false);
  const [chatLog, setChatLog] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState<string>('');
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const [currentImageData, setCurrentImageData] = useState<ImageData>({ base64: null, mimeType: null });

  // --- Refs ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null); 
  const chatEndRef = useRef<HTMLDivElement>(null);
  const requestRef = useRef<number | null>(null);

  // --- API Key ---
  const API_KEY = "AIzaSyBLsOuHlVUHAB67k1hDL5Z_GZV0Ok1JCi8"; // Paste your key here

  // --- Initialization ---
  useEffect(() => {
    async function loadModel() {
      try {
        const loadedModel = await mobilenet.load();
        setModel(loadedModel);
        setLoadingModel(false);
      } catch (err) {
        console.error('Error loading model:', err);
        setLoadingModel(false);
      }
    }
    loadModel();
    return () => stopWebcam();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatLog, isThinking]);

  // --- Helpers ---
  const resetUI = () => {
    setImageSrc(null);
    setPredictions([]);
    setChatLog([]);
    setCurrentImageData({ base64: null, mimeType: null });
  };

  const handleTabSwitch = (tab: 'upload' | 'webcam') => {
    stopWebcam();
    setActiveTab(tab);
    resetUI();
  };

  // --- Logic ---
  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result as string;
      setImageSrc(result);
      const base64 = result.split(',')[1];
      const mimeType = result.split(';')[0].split(':')[1];
      setCurrentImageData({ base64, mimeType });
      setTimeout(() => { if (imageRef.current) classifyImage(imageRef.current); }, 100);
    };
    reader.readAsDataURL(file);
  };

  const startWebcam = async () => {
    if (!model) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            setIsWebcamActive(true);
        };
      }
    } catch (err) {
      alert("Could not access webcam.");
    }
  };

  const stopWebcam = () => {
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);
  };

  const predictWebcam = async () => {
    if (videoRef.current && videoRef.current.readyState === 4 && isWebcamActive) {
      await classifyImage(videoRef.current, true);
      requestRef.current = requestAnimationFrame(predictWebcam);
    }
  };

  useEffect(() => {
    if (isWebcamActive) predictWebcam();
    else if(requestRef.current) cancelAnimationFrame(requestRef.current);
  }, [isWebcamActive]);

  const captureWebcam = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (ctx) {
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        ctx.drawImage(videoRef.current, 0, 0);
        const dataUrl = canvasRef.current.toDataURL('image/png');
        stopWebcam();
        setImageSrc(dataUrl);
        setCurrentImageData({ base64: dataUrl.split(',')[1], mimeType: dataUrl.split(';')[0].split(':')[1] });
        setTimeout(() => { if (imageRef.current) classifyImage(imageRef.current); }, 100);
    }
  };

  const classifyImage = async (element: HTMLVideoElement | HTMLImageElement, isLive = false) => {
    if (!model || !element) return;
    try {
      const results = await model.classify(element);
      if (results && results.length > 0) setPredictions(results.slice(0, 3));
    } catch (err) {
      if (!isLive) console.error(err);
    }
  };

  const handleQaSubmit = async () => {
    if (!chatInput.trim() || !currentImageData.base64) return;
    const question = chatInput.trim();
    setChatLog(prev => [...prev, { role: 'user', text: question }]);
    setChatInput('');
    setIsThinking(true);

    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${API_KEY}`;
    const payload = { contents: [{ role: "user", parts: [{ text: question }, { inlineData: { mimeType: currentImageData.mimeType, data: currentImageData.base64 } }] }] };

    try {
        const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const data = await response.json();
        const answer = data.candidates?.[0]?.content?.parts?.[0]?.text || "Sorry, I couldn't understand the image.";
        setChatLog(prev => [...prev, { role: 'ai', text: answer }]);
    } catch (err) {
        setChatLog(prev => [...prev, { role: 'ai', text: "Connection error." }]);
    } finally {
        setIsThinking(false);
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQaSubmit();
    }
  };

  return (
    // Outer container: Full viewport height (h-screen)
    <div className="h-screen w-screen bg-gray-100 flex items-center justify-center p-4 font-sans text-gray-800 overflow-hidden">
      
      {/* Main Card: 95% width, 90% height */}
      <div className="w-[95%] h-[90vh] bg-white rounded-3xl shadow-2xl overflow-hidden flex flex-col md:flex-row border border-gray-200">
        
        {/* Left Side: Media & Controls */}
        <div className="w-full md:w-1/2 p-6 border-b md:border-b-0 md:border-r border-gray-200 flex flex-col h-full bg-white">
            
            {/* Header */}
            <div className="flex items-center space-x-4 mb-6 flex-shrink-0">
                <div className="p-3 bg-blue-100 rounded-2xl">
                    <IconRobot />
                </div>
                <h1 className="text-3xl font-bold text-gray-900 tracking-tight">VisionAI</h1>
            </div>

            {/* NAVIGATION TABS - UPDATED */}
            <div className="flex bg-slate-200 p-1.5 rounded-2xl mb-6 flex-shrink-0">
                <button 
                    onClick={() => handleTabSwitch('upload')}
                    // IF SELECTED: Blue Background, White Text
                    // IF NOT SELECTED: White Background, Dark Text
                    className={`flex-1 py-3 text-sm font-bold rounded-xl transition-all duration-200 ${
                        activeTab === 'upload' 
                        ? 'bg-blue-600 text-white shadow-md' 
                        : 'bg-white text-slate-700 shadow-sm hover:bg-gray-50'
                    }`}
                >
                    Upload Image
                </button>
                <div className="w-2"></div> {/* Spacing between buttons */}
                <button 
                    onClick={() => handleTabSwitch('webcam')}
                    // IF SELECTED: Blue Background, White Text
                    // IF NOT SELECTED: White Background, Dark Text
                    className={`flex-1 py-3 text-sm font-bold rounded-xl transition-all duration-200 ${
                        activeTab === 'webcam' 
                        ? 'bg-blue-600 text-white shadow-md' 
                        : 'bg-white text-slate-700 shadow-sm hover:bg-gray-50'
                    }`}
                >
                    Live Webcam
                </button>
            </div>

            {/* Content Area - Flex Grow to fill remaining height */}
            <div className="flex-grow flex flex-col justify-center bg-gray-50 rounded-3xl border-2 border-dashed border-gray-200 relative overflow-hidden group hover:border-blue-300 transition-colors">
                
                {/* Loader */}
                {loadingModel && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/90 z-50">
                        <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-200 border-t-blue-600 mb-4"></div>
                        <p className="text-gray-600 font-medium">Loading AI Models...</p>
                    </div>
                )}

                {/* Upload Tab */}
                {activeTab === 'upload' && !imageSrc && (
                    <label className="flex flex-col items-center justify-center cursor-pointer h-full w-full p-6 hover:bg-blue-50/30 transition-colors">
                        <div className="bg-white p-6 rounded-full shadow-sm mb-4">
                            <IconUpload />
                        </div>
                        <span className="text-xl text-gray-700 font-bold">Drag & drop image here</span>
                        <span className="text-gray-500 mt-2">or click to browse files</span>
                        <input type="file" className="hidden" accept="image/*" onChange={handleFileUpload} />
                    </label>
                )}

                {/* Webcam Tab */}
                {activeTab === 'webcam' && !isWebcamActive && !imageSrc && (
                    <div className="flex flex-col items-center justify-center h-full p-6">
                         <p className="text-gray-500 mb-6 text-lg">Use your camera for real-time AI analysis</p>
                         <button onClick={startWebcam} className="flex items-center px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-full font-bold text-lg transition-transform transform hover:scale-105 shadow-xl">
                            <IconCamera /> Activate Camera
                         </button>
                    </div>
                )}

                {/* Image/Video Display */}
                <img ref={imageRef} src={imageSrc || '#'} alt="Target" className={`w-full h-full object-contain p-2 ${imageSrc ? 'block' : 'hidden'}`} crossOrigin="anonymous" />
                <video ref={videoRef} className={`w-full h-full object-cover absolute inset-0 rounded-3xl ${isWebcamActive ? 'block' : 'hidden'}`} playsInline muted />
                <canvas ref={canvasRef} className="hidden" />

                {/* Webcam Controls */}
                {isWebcamActive && (
                    <div className="absolute bottom-8 left-0 right-0 flex justify-center space-x-6 z-20">
                        <button onClick={captureWebcam} className="bg-white text-blue-600 px-8 py-3 rounded-full font-bold shadow-lg hover:bg-gray-50 transition-colors ring-4 ring-blue-500/20">Capture Photo</button>
                        <button onClick={stopWebcam} className="bg-red-500 text-white px-8 py-3 rounded-full font-bold shadow-lg hover:bg-red-600 transition-colors">Stop</button>
                    </div>
                )}
            </div>
        </div>

        {/* Right Side: Results & Chat */}
        <div className="w-full md:w-1/2 flex flex-col h-full bg-gray-50">
            
            {/* Predictions Panel - Fixed Height */}
            <div className="p-6 border-b border-gray-200 bg-white flex-shrink-0">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4">Real-time Analysis</h3>
                <div className="grid grid-cols-1 gap-3">
                    {predictions.length === 0 && (
                        <div className="text-gray-400 text-sm italic py-2">Waiting for image...</div>
                    )}
                    {predictions.map((p, i) => (
                        <div key={i} className="flex items-center justify-between bg-gray-50 p-3 rounded-xl border border-gray-100">
                            <span className="font-semibold text-gray-700 capitalize">{p.className}</span>
                            <div className="flex items-center space-x-3">
                                <div className="w-24 bg-gray-200 rounded-full h-2">
                                    <div className="bg-blue-500 h-2 rounded-full transition-all duration-500" style={{ width: `${p.probability * 100}%` }}></div>
                                </div>
                                <span className="text-blue-600 font-bold w-12 text-right">{(p.probability * 100).toFixed(0)}%</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Chat Panel - Flex Grow to fill remaining height */}
            <div className="flex-grow flex flex-col p-6 overflow-hidden relative">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4 flex-shrink-0">AI Assistant</h3>
                
                {/* Chat Log - Scrollable */}
                <div className="flex-grow overflow-y-auto space-y-4 mb-4 pr-2 custom-scrollbar">
                    {chatLog.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full opacity-40">
                            <p className="text-6xl mb-4">💬</p>
                            <p className="text-gray-600 font-semibold text-lg">Chat with your Image</p>
                            <p className="text-gray-500">Upload an image to start asking questions.</p>
                        </div>
                    )}
                    {chatLog.map((msg, idx) => (
                        <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[85%] p-4 rounded-2xl text-sm leading-relaxed shadow-sm ${
                                msg.role === 'user' ? 'bg-blue-600 text-white rounded-br-sm' : 'bg-white text-gray-800 border border-gray-200 rounded-bl-sm'
                            }`}>
                                {msg.text}
                            </div>
                        </div>
                    ))}
                    {isThinking && (
                        <div className="flex justify-start">
                             <div className="bg-white border border-gray-200 text-gray-500 px-5 py-3 rounded-2xl rounded-bl-sm text-sm flex items-center shadow-sm">
                                <span className="flex space-x-1 mr-2">
                                    <span className="animate-bounce">●</span><span className="animate-bounce delay-75">●</span><span className="animate-bounce delay-150">●</span>
                                </span>
                                Thinking...
                             </div>
                        </div>
                    )}
                    <div ref={chatEndRef}></div>
                </div>

                {/* Input Area - Pinned to bottom */}
                <div className="relative flex-shrink-0 pt-2">
                    <input 
                        type="text" 
                        value={chatInput}
                        onChange={(e) => setChatInput(e.target.value)}
                        onKeyDown={handleKeyPress}
                        placeholder={imageSrc ? "Ask a question about the image..." : "Upload an image to start chatting..."}
                        disabled={!imageSrc || isThinking}
                        className="w-full bg-white border border-gray-300 rounded-2xl py-4 pl-6 pr-14 text-gray-800 placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10 disabled:bg-gray-100 transition-all shadow-sm"
                    />
                    <button 
                        onClick={handleQaSubmit}
                        disabled={!imageSrc || isThinking}
                        className="absolute right-3 top-5 p-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:bg-gray-300 disabled:text-gray-500 transition-colors shadow-sm"
                    >
                        <IconSend />
                    </button>
                </div>
            </div>
        </div>
      </div>
    </div>
  );
}

export default App;