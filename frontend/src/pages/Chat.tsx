import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";

// ── Config ─────────────────────────────────────────────────────────────────
// All API calls go through the Vite proxy (/api → localhost:8502) so the
// browser only ever talks to the same host:port it loaded the page from.
// This means the network URL (e.g. 192.168.1.x:3000) works out-of-the-box
// on any device without CORS or firewall issues on port 8502.
const API_URL = "";

// ── Language list (Flores-200 codes) ──────────────────────────────────────
const LANGUAGES = [
  { code: "eng_Latn", label: "English",           native: "English",      rtl: false },
  { code: "hin_Deva", label: "Hindi",             native: "हिन्दी",       rtl: false },
  { code: "ben_Beng", label: "Bengali",           native: "বাংলা",        rtl: false },
  { code: "tam_Taml", label: "Tamil",             native: "தமிழ்",       rtl: false },
  { code: "tel_Telu", label: "Telugu",            native: "తెలుగు",      rtl: false },
  { code: "mal_Mlym", label: "Malayalam",         native: "മലയാളം",      rtl: false },
  { code: "kan_Knda", label: "Kannada",           native: "ಕನ್ನಡ",       rtl: false },
  { code: "mar_Deva", label: "Marathi",           native: "मराठी",       rtl: false },
  { code: "guj_Gujr", label: "Gujarati",          native: "ગુજરાતી",     rtl: false },
  { code: "pan_Guru", label: "Punjabi",           native: "ਪੰਜਾਬੀ",     rtl: false },
  { code: "urd_Arab", label: "Urdu",              native: "اردو",        rtl: true  },
  { code: "ory_Orya", label: "Odia",              native: "ଓଡ଼ିଆ",       rtl: false },
  { code: "asm_Beng", label: "Assamese",          native: "অসমীয়া",     rtl: false },
  { code: "mai_Deva", label: "Maithili",          native: "मैथिली",     rtl: false },
  { code: "sat_Olck", label: "Santali",           native: "ᱥᱟᱱᱛᱟᱲᱤ",   rtl: false },
  { code: "kas_Arab", label: "Kashmiri (Ar.)",    native: "کٲشُر",       rtl: true  },
  { code: "kas_Deva", label: "Kashmiri (Dev.)",   native: "कॉशुर",      rtl: false },
  { code: "npi_Deva", label: "Nepali",            native: "नेपाली",     rtl: false },
  { code: "mni_Beng", label: "Manipuri (Bn.)",    native: "মৈতেই",       rtl: false },
  { code: "mni_Mtei", label: "Manipuri (Mt.)",    native: "ꯃꯩꯇꯩꯂꯣꯟ",   rtl: false },
  { code: "doi_Deva", label: "Dogri",             native: "डोगरी",      rtl: false },
  { code: "brx_Deva", label: "Bodo",              native: "बड़ो",        rtl: false },
  { code: "gom_Deva", label: "Konkani",           native: "कोंकणी",    rtl: false },
  { code: "san_Deva", label: "Sanskrit",          native: "संस्कृतम्",  rtl: false },
  { code: "snd_Arab", label: "Sindhi (Ar.)",      native: "سنڌي",       rtl: true  },
  { code: "snd_Deva", label: "Sindhi (Dev.)",     native: "सिन्धी",    rtl: false },
];

const PHASES = ["RAPPORT", "MOOD", "SLEEP", "ENERGY", "APPETITE", "CONCENTRATION"];

const DOMAINS = [
  { key: "sleep",          label: "Sleep" },
  { key: "mood",           label: "Mood / Anhedonia" },
  { key: "energy",         label: "Energy" },
  { key: "appetite",       label: "Appetite" },
  { key: "concentration",  label: "Concentration" },
];

const ABOUT_ITEMS = [
  "gpt-4o-mini / gpt-4o (LLM + CoT scoring)",
  "LangGraph (dialogue flow)",
  "SBERT (safety monitor)",
  "IndicTrans2 (Indian language translation)",
  "Whisper (voice-to-text)",
];

type Message = { id: number; role: "agent" | "user"; text: string; ts: Date };

type Confidence = Record<string, number>;   // domain → 0-100

// ── UUID that works in both secure (HTTPS/localhost) and plain HTTP contexts ──
function makeUUID(): string {
  // crypto.randomUUID() requires a secure context; fall back when unavailable
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // RFC-4122 v4 fallback using Math.random
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

// ── Generate a stable session id per browser tab ───────────────────────────
function getSessionId(): string {
  const key = "mh_session_id";
  let id = sessionStorage.getItem(key);
  if (!id) { id = makeUUID(); sessionStorage.setItem(key, id); }
  return id;
}

export default function Chat() {
  const [, setLocation] = useLocation();

  // Read lang + mood from URL params set by the landing page
  const urlParams = new URLSearchParams(window.location.search);
  const initLangCode = urlParams.get("lang") ?? "eng_Latn";
  const initMood     = urlParams.get("mood") ?? "";

  const [sessionId]        = useState(getSessionId);
  const [resetKey,         setResetKey]         = useState(0);   // bumped on reset
  const [activeLangCode,   setActiveLangCode]   = useState(initLangCode);
  const [messages,         setMessages]         = useState<Message[]>([]);
  const [input,            setInput]            = useState("");
  const [isTyping,         setIsTyping]         = useState(false);
  const [isRecording,      setIsRecording]      = useState(false);
  const [recordingTime,    setRecordingTime]    = useState(0);
  const [phaseIndex,       setPhaseIndex]       = useState(0);
  const [turn,             setTurn]             = useState(0);
  const [confidence,       setConfidence]       = useState<Confidence>(
    Object.fromEntries(DOMAINS.map(d => [d.key, 0]))
  );
  const [domainsAssessed,  setDomainsAssessed]  = useState<string[]>([]);
  const [safetyAlert,      setSafetyAlert]      = useState(false);
  const [sessionDone,      setSessionDone]      = useState(false);
  const [apiError,         setApiError]         = useState("");

  // ── Sidebar toggle (hidden on mobile by default) ─────────────────────
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // ── Intro overlay + typewriter ──────────────────────────────────────────
  const [showIntro,      setShowIntro]      = useState(true);
  const [introVisible,   setIntroVisible]   = useState(false); // animates in
  const [typewriterText, setTypewriterText] = useState("");
  const [isTypewriting,  setIsTypewriting]  = useState(false);

  const chatEndRef  = useRef<HTMLDivElement>(null);
  const inputRef    = useRef<HTMLInputElement>(null);
  const timerRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const waveRef     = useRef<HTMLCanvasElement>(null);
  const waveAnimRef = useRef<number>(0);
  const mediaRecRef = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);

  const activeLang = LANGUAGES.find(l => l.code === activeLangCode) || LANGUAGES[0];

  // Auto-scroll to latest message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping, typewriterText]);

  // ── Typewriter helper ─────────────────────────────────────────────────
  const runTypewriter = useCallback((text: string, onDone: () => void) => {
    setTypewriterText("");
    setIsTypewriting(true);
    let i = 0;
    const interval = setInterval(() => {
      i++;
      setTypewriterText(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(interval);
        setIsTypewriting(false);
        onDone();
      }
    }, 28);
  }, []);

  // ── Fetch opening greeting from API ───────────────────────────────────
  const fetchGreeting = useCallback(() => {
    setIsTyping(false); // use typewriter instead of typing dots for greeting
    const moodLabels: Record<string, string> = {
      "0": "very low", "1": "struggling", "2": "neutral", "3": "okay", "4": "good",
    };
    const moodLabel = moodLabels[initMood] ?? "";

    fetch(`${API_URL}/api/greeting`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, lang_code: activeLangCode, mood: moodLabel }),
    })
      .then(r => r.json())
      .then(data => {
        const reply = data.reply || "Hi, I'm glad you're here. How have things been going for you lately?";
        runTypewriter(reply, () => {
          setMessages([{ id: Date.now(), role: "agent", text: reply, ts: new Date() }]);
          setTypewriterText("");
        });
        _syncState(data);
      })
      .catch(() => {
        const fallback = "Hi, I'm glad you're here. How have things been going for you lately?";
        runTypewriter(fallback, () => {
          setMessages([{ id: Date.now(), role: "agent", text: fallback, ts: new Date() }]);
          setTypewriterText("");
        });
        setApiError("Could not reach the chat server. Make sure the backend is running.");
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, activeLangCode, initMood, runTypewriter]);

  // ── Show intro overlay on mount and after every reset ─────────────────
  useEffect(() => {
    setShowIntro(true);
    setIntroVisible(false);
    setMessages([]);
    setTypewriterText("");
    setIsTypewriting(false);
    const t = setTimeout(() => setIntroVisible(true), 50);
    return () => clearTimeout(t);
  }, [resetKey]);

  // ── Dismiss intro, then start the greeting typewriter ─────────────────
  const dismissIntro = useCallback(() => {
    setIntroVisible(false);
    setTimeout(() => {
      setShowIntro(false);
      fetchGreeting();
    }, 500);
  }, [fetchGreeting]);

  // ── Sync phase / scores from API response ─────────────────────────────────
  function _syncState(data: {
    phase?: string; turn?: number; confidence?: Confidence;
    domains_assessed?: string[]; safety_alert?: boolean;
    session_complete?: boolean; quota_error?: boolean;
  }) {
    if (data.phase) {
      const idx = PHASES.indexOf(data.phase);
      if (idx >= 0) setPhaseIndex(idx);
    }
    if (typeof data.turn === "number")   setTurn(data.turn);
    if (data.confidence)                 setConfidence(data.confidence);
    if (data.domains_assessed)           setDomainsAssessed(data.domains_assessed);
    if (data.safety_alert)               setSafetyAlert(true);
    if (data.session_complete)           setSessionDone(true);
    if (data.quota_error)                setApiError("⚠️ LLM quota error — check your API key.");
  }

  // ── Send text message ──────────────────────────────────────────────────────
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isTyping || sessionDone) return;

    setMessages(prev => [...prev, { id: Date.now(), role: "user", text, ts: new Date() }]);
    setInput("");
    setIsTyping(true);
    setApiError("");

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId, lang_code: activeLangCode }),
      });
      const data = await res.json();
      setIsTyping(false);
      if (data.reply) {
        setMessages(prev => [...prev, { id: Date.now(), role: "agent", text: data.reply, ts: new Date() }]);
        _syncState(data);
      }
    } catch {
      setIsTyping(false);
      setApiError("Network error — make sure the backend is running on port 8502.");
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input, isTyping, sessionDone, sessionId, activeLangCode]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  // ── Reset session ──────────────────────────────────────────────────────────
  const handleReset = async () => {
    await fetch(`${API_URL}/api/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, lang_code: activeLangCode }),
    }).catch(() => {});

    setInput("");
    setTurn(0);
    setPhaseIndex(0);
    setConfidence(Object.fromEntries(DOMAINS.map(d => [d.key, 0])));
    setDomainsAssessed([]);
    setIsTyping(false);
    setIsRecording(false);
    setRecordingTime(0);
    setSafetyAlert(false);
    setSessionDone(false);
    setApiError("");
    setResetKey(k => k + 1);  // triggers re-show of intro overlay
  };

  // ── Voice recording with Web MediaRecorder API ─────────────────────────────
  const stopWaveAnimation = useCallback(() => cancelAnimationFrame(waveAnimRef.current), []);

  const startWaveAnimation = useCallback(() => {
    const canvas = waveRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const draw = () => {
      const { width: w, height: h } = canvas;
      ctx.clearRect(0, 0, w, h);
      const bars = 40;
      const barW = w / bars;
      for (let i = 0; i < bars; i++) {
        const amp = Math.random() * (h * 0.7) + h * 0.1;
        const x = i * barW + barW / 2;
        const g = ctx.createLinearGradient(x, h / 2 - amp / 2, x, h / 2 + amp / 2);
        g.addColorStop(0, "rgba(245,158,11,0.9)");
        g.addColorStop(1, "rgba(20,184,166,0.6)");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.roundRect(x - barW * 0.3, h / 2 - amp / 2, barW * 0.6, amp, 3);
        ctx.fill();
      }
      waveAnimRef.current = requestAnimationFrame(draw);
    };
    draw();
  }, []);

  useEffect(() => {
    if (isRecording) { startWaveAnimation(); timerRef.current = setInterval(() => setRecordingTime(t => t + 1), 1000); }
    else             { stopWaveAnimation(); if (timerRef.current) clearInterval(timerRef.current); }
    return () => { stopWaveAnimation(); if (timerRef.current) clearInterval(timerRef.current); };
  }, [isRecording, startWaveAnimation, stopWaveAnimation]);

  const formatTime = (s: number) =>
    `${Math.floor(s / 60).toString().padStart(2, "0")}:${(s % 60).toString().padStart(2, "0")}`;

  const toggleRecording = async () => {
    if (isRecording) {
      // Stop recording
      mediaRecRef.current?.stop();
      setIsRecording(false);
      setRecordingTime(0);
    } else {
      // Start recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks.current = [];
        const mr = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
        mediaRecRef.current = mr;

        mr.ondataavailable = e => { if (e.data.size > 0) audioChunks.current.push(e.data); };

        mr.onstop = async () => {
          stream.getTracks().forEach(t => t.stop());
          const blob = new Blob(audioChunks.current, { type: "audio/webm" });
          // Add placeholder while transcribing
          const placeholderId = Date.now();
          setMessages(prev => [...prev, { id: placeholderId, role: "user", text: "🎤 Transcribing…", ts: new Date() }]);
          setIsTyping(true);
          setApiError("");

          try {
            const fd = new FormData();
            fd.append("audio", blob, "recording.webm");
            fd.append("lang_code", activeLangCode);
            const res  = await fetch(`${API_URL}/api/transcribe`, { method: "POST", body: fd });
            const data = await res.json();
            const transcript: string = (data.text || "").trim();

            if (!transcript) {
              // Nothing usable — remove placeholder silently and show a soft hint
              setMessages(prev => prev.filter(m => m.id !== placeholderId));
              setIsTyping(false);
              setApiError("Couldn't catch that — try speaking a bit closer to the mic and try again.");
              // Clear the hint after 4 seconds so it doesn't linger
              setTimeout(() => setApiError(""), 4000);
              return;
            }

            // Replace placeholder with the real transcript
            setMessages(prev => prev.map(m =>
              m.id === placeholderId ? { ...m, text: `🎤 ${transcript}` } : m
            ));

            // Send transcript to backend as a regular chat message
            const chatRes  = await fetch(`${API_URL}/api/chat`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ message: transcript, session_id: sessionId, lang_code: activeLangCode }),
            });
            const chatData = await chatRes.json();
            setIsTyping(false);
            if (chatData.reply) {
              setMessages(prev => [...prev, { id: Date.now(), role: "agent", text: chatData.reply, ts: new Date() }]);
              _syncState(chatData);
            }
          } catch {
            setIsTyping(false);
            setMessages(prev => prev.map(m => m.id === placeholderId ? { ...m, text: "🎤 (transcription failed)" } : m));
            setApiError("Transcription failed — make sure Whisper is installed (pip install openai-whisper).");
          }
        };

        mr.start();
        setIsRecording(true);
      } catch {
        setApiError("Microphone access denied — please allow microphone access in your browser.");
      }
    }
  };

  const phaseTurnsTotal = 4;
  const phaseTurnsDone  = Math.min(turn % phaseTurnsTotal, phaseTurnsTotal);

  return (
    <div className="h-screen flex flex-col bg-background text-foreground font-sans overflow-hidden">

      {/* ── Top bar ── */}
      <header className="flex items-center gap-3 px-4 py-3 border-b border-border/60 bg-card/60 backdrop-blur-md shrink-0 z-20">
        {/* Hamburger — mobile only */}
        <button
          onClick={() => setSidebarOpen(o => !o)}
          className="md:hidden text-muted-foreground hover:text-foreground transition-colors p-1"
          aria-label="Toggle settings"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/>
          </svg>
        </button>
        <button
          onClick={() => setLocation("/")}
          className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors text-sm"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15 18-6-6 6-6"/></svg>
          <span className="hidden sm:inline">Back</span>
        </button>
        <div className="flex items-center gap-2">
          <img
            src="/therapist-avatar.png"
            alt="Dr. UB"
            className="w-7 h-7 rounded-full object-cover border border-secondary/40"
          />
          <span className="font-semibold text-foreground">Dr. UB</span>
        </div>
        <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground border border-border/50 px-3 py-1 rounded-full bg-card/40">
          {activeLang.native}
        </div>
      </header>

      {/* ── 3-column body ── */}
      <div className="flex flex-1 min-h-0 overflow-hidden relative">

        {/* Mobile sidebar backdrop */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-40 bg-black/40 md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* ── LEFT SIDEBAR ── */}
        <aside className={`
          flex flex-col gap-4 p-4 border-r border-border/50 bg-sidebar/90 backdrop-blur-md overflow-y-auto
          md:relative md:w-64 md:shrink-0 md:translate-x-0 md:flex
          fixed inset-y-0 left-0 z-50 w-72 transition-transform duration-300
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
        `}>

          {/* Close button — mobile only */}
          <div className="flex items-center justify-between md:hidden mb-1">
            <span className="text-xs font-semibold text-muted-foreground uppercase tracking-widest">Settings</span>
            <button onClick={() => setSidebarOpen(false)} className="text-muted-foreground hover:text-foreground p-1">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg>
            </button>
          </div>

          <div className="hidden md:flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-widest pt-1">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
            Settings
          </div>

          {/* Language selector */}
          <div className="glass-card rounded-xl p-3 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-widest">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
              Language
            </div>
            <p className="text-[11px] text-muted-foreground">Respond in:</p>
            <div className="relative">
              <select
                value={activeLangCode}
                onChange={e => setActiveLangCode(e.target.value)}
                className="w-full bg-muted/60 border border-border/60 rounded-lg px-3 py-2 text-sm text-foreground appearance-none cursor-pointer focus:outline-none focus:ring-1 focus:ring-primary/50"
              >
                {LANGUAGES.map(l => (
                  <option key={l.code} value={l.code}>{l.label}</option>
                ))}
              </select>
              <div className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6"/></svg>
              </div>
            </div>
          </div>

          {/* Voice info */}
          <div className="glass-card rounded-xl p-3 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-widest">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>
              Voice Input
            </div>
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              Uses OpenAI Whisper (runs locally). Supports Hindi, Urdu, English and 90+ languages.
            </p>
            <div className="flex items-start gap-2 text-[11px] text-secondary">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" className="mt-0.5 shrink-0"><path d="M20 6L9 17l-5-5"/></svg>
              <span>Use the mic button below to record your voice.</span>
            </div>
          </div>

          {/* About */}
          <div className="glass-card rounded-xl p-3 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-widest">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18M9 21V9"/></svg>
              About
            </div>
            <p className="text-[11px] text-muted-foreground">Mental health screening powered by:</p>
            <ul className="space-y-1.5">
              {ABOUT_ITEMS.map((item, i) => (
                <li key={i} className="flex items-start gap-2 text-[11px] text-muted-foreground">
                  <span className="text-primary mt-0.5 shrink-0">•</span>
                  <span dangerouslySetInnerHTML={{ __html: item.replace(/\(([^)]+)\)/g, '<span class="text-foreground/70">($1)</span>') }} />
                </li>
              ))}
            </ul>
            <div className="flex items-start gap-1.5 pt-1 text-[11px] text-amber-400/80">
              <span className="shrink-0">⚠</span>
              <span>This is a screening tool only — not a clinical diagnosis.</span>
            </div>
          </div>

          <div className="mt-auto pt-2">
            <button
              onClick={handleReset}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl border border-border/60 bg-card/40 hover:bg-destructive/10 hover:border-destructive/40 text-sm text-muted-foreground hover:text-destructive transition-all"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
              Reset Session
            </button>
          </div>
        </aside>

        {/* ── CENTER CHAT ── */}
        <main className="flex flex-col flex-1 min-w-0 relative">

          {/* ── Intro overlay ── */}
          {showIntro && (
            <div
              className={`absolute inset-0 z-30 flex items-center justify-center bg-background/80 backdrop-blur-sm transition-all duration-500 ${
                introVisible ? "opacity-100" : "opacity-0 pointer-events-none"
              }`}
            >
              <div
                className={`glass-card rounded-2xl p-8 max-w-sm w-full mx-6 flex flex-col items-center text-center gap-5 shadow-2xl border border-primary/20 transition-all duration-500 ${
                  introVisible ? "translate-y-0 scale-100" : "translate-y-8 scale-95"
                }`}
              >
                <img
                  src="/therapist-avatar.png"
                  alt="Dr. UB"
                  className="w-24 h-24 rounded-full object-cover border-2 border-secondary/50 shadow-lg"
                />
                <h2 className="text-2xl font-serif text-foreground">Dr. UB</h2>
                <p className="text-xs text-secondary font-medium tracking-wide uppercase">
                  Mental Health Screening Agent
                </p>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Hello, I'm Dr. UB. This is a safe, confidential space. I'll ask you a few gentle
                  questions about how you've been feeling lately — there are no right or wrong answers.
                </p>
                <button
                  onClick={dismissIntro}
                  className="w-full py-3 rounded-xl font-medium text-sm text-primary-foreground transition-opacity hover:opacity-90"
                  style={{ background: "linear-gradient(135deg, hsl(33 95% 55%), hsl(33 95% 45%))" }}
                >
                  Begin Session
                </button>
                <p className="text-[10px] text-muted-foreground/50">No sign-up needed · Free · Private</p>
              </div>
            </div>
          )}

          {/* Chat sub-header */}
          <div className="px-6 py-4 border-b border-border/40 bg-background/60 backdrop-blur-sm shrink-0">
            <div className="flex items-center gap-3">
              <img
                src="/therapist-avatar.png"
                alt="Therapist avatar"
                className="w-10 h-10 rounded-full object-cover border-2 border-secondary/40 shadow-md shadow-secondary/20"
              />
              <div>
                <h1 className="text-xl font-serif text-foreground">Dr. UB</h1>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Mental Health Screening Agent · Powered by GPT-4o + LangGraph
                </p>
              </div>
            </div>
          </div>

          {/* Safety alert banner */}
          {safetyAlert && (
            <div className="mx-6 mt-3 px-4 py-3 rounded-xl bg-destructive/15 border border-destructive/40 text-sm text-destructive flex items-start gap-2">
              <span className="text-xl shrink-0">🆘</span>
              <div>
                <p className="font-semibold">We're concerned about your safety.</p>
                <p className="text-xs mt-0.5 text-destructive/80">
                  If you're in immediate danger, please call the iCall helpline: <strong>9152987821</strong> or emergency services <strong>112</strong>.
                </p>
              </div>
            </div>
          )}

          {/* API error banner */}
          {apiError && (
            <div className="mx-6 mt-2 px-4 py-2 rounded-xl bg-yellow-500/10 border border-yellow-500/30 text-xs text-yellow-400 flex items-center gap-2">
              <span>⚠️</span> {apiError}
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
            {messages.map(msg => (
              <div
                key={msg.id}
                className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
                dir={activeLang.rtl ? "rtl" : "ltr"}
              >
                {msg.role === "agent" && (
                  <img
                    src="/therapist-avatar.png"
                    alt="Dr. UB"
                    className="shrink-0 w-9 h-9 rounded-full object-cover border border-secondary/30 shadow-sm"
                  />
                )}
                <div className={`max-w-[72%] ${msg.role === "user" ? "items-end" : "items-start"} flex flex-col gap-1`}>
                  <div
                    className={`px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm ${
                      msg.role === "agent"
                        ? "glass-card rounded-tl-sm text-foreground"
                        : "rounded-tr-sm text-primary-foreground"
                    }`}
                    style={msg.role === "user" ? {
                      background: "linear-gradient(135deg, hsl(33 95% 55%), hsl(33 95% 45%))",
                    } : undefined}
                  >
                    {msg.text}
                  </div>
                  <span className="text-[10px] text-muted-foreground/60 px-1">
                    {msg.ts.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </span>
                </div>
              </div>
            ))}

            {/* Typewriter bubble for greeting */}
            {(isTypewriting || typewriterText) && (
              <div className="flex gap-3">
                <img
                  src="/therapist-avatar.png"
                  alt="Dr. UB"
                  className="shrink-0 w-9 h-9 rounded-full object-cover border border-secondary/30 shadow-sm"
                />
                <div className="px-4 py-3 rounded-2xl rounded-tl-sm text-sm leading-relaxed shadow-sm glass-card text-foreground">
                  {typewriterText}
                  {isTypewriting && (
                    <span className="inline-block w-0.5 h-4 bg-secondary align-middle ml-0.5 animate-pulse" />
                  )}
                </div>
              </div>
            )}

            {isTyping && (
              <div className="flex gap-3">
                <img
                  src="/therapist-avatar.png"
                  alt="Dr. UB"
                  className="shrink-0 w-9 h-9 rounded-full object-cover border border-secondary/30 shadow-sm"
                />
                <div className="glass-card rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-1.5">
                  {[0, 1, 2].map(i => (
                    <span key={i} className="w-2 h-2 rounded-full bg-muted-foreground/60 animate-bounce" style={{ animationDelay: `${i * 150}ms` }} />
                  ))}
                </div>
              </div>
            )}

            {/* Session complete message */}
            {sessionDone && !isTyping && (
              <div className="flex justify-center">
                <div className="glass-card rounded-2xl px-6 py-4 text-center space-y-3 max-w-sm">
                  <p className="text-sm font-medium text-foreground">✅ Screening complete</p>
                  <p className="text-xs text-muted-foreground">See your domain scores in the panel on the right.</p>
                  <button onClick={handleReset} className="px-4 py-2 rounded-full bg-primary text-primary-foreground text-xs hover:bg-primary/90 transition">
                    Start New Session
                  </button>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* Voice Recorder Bar */}
          <div className="px-6 pb-2 shrink-0">
            <div className="glass-card rounded-xl p-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>
                  Voice Input
                </div>
              </div>
              <p className="text-[11px] text-muted-foreground mb-2">
                {isRecording
                  ? "Recording… click stop when done."
                  : "Record your voice — Whisper detects your language automatically."}
              </p>
              <div className="flex items-center gap-3 bg-muted/30 rounded-lg px-3 py-2">
                <button
                  onClick={toggleRecording}
                  disabled={sessionDone}
                  className={`shrink-0 w-9 h-9 rounded-full flex items-center justify-center transition-all disabled:opacity-30 ${
                    isRecording
                      ? "bg-destructive/20 border border-destructive/50 text-destructive animate-pulse"
                      : "bg-primary/15 border border-primary/30 text-primary hover:bg-primary/25"
                  }`}
                >
                  {isRecording
                    ? <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
                    : <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>
                  }
                </button>
                <canvas
                  ref={waveRef}
                  width={260} height={36}
                  className={`flex-1 rounded transition-opacity ${isRecording ? "opacity-100" : "opacity-20"}`}
                  style={{ imageRendering: "pixelated" }}
                />
                <span className="text-xs font-mono text-muted-foreground shrink-0 w-10 text-right">
                  {formatTime(recordingTime)}
                </span>
              </div>
            </div>
          </div>

          {/* Text Input */}
          <div className="px-6 pb-4 shrink-0">
            <div className={`flex items-center gap-2 glass-card rounded-xl px-4 py-2 ${sessionDone ? "opacity-50" : ""}`}>
              <input
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={sessionDone ? "Session complete — reset to start again" : "Type your message here…"}
                disabled={sessionDone}
                dir={activeLang.rtl ? "rtl" : "ltr"}
                className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground/50 outline-none disabled:cursor-not-allowed"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isTyping || sessionDone}
                className="shrink-0 w-8 h-8 rounded-lg flex items-center justify-center bg-primary/15 border border-primary/30 text-primary hover:bg-primary hover:text-primary-foreground transition-all disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>
              </button>
            </div>
          </div>
        </main>

        {/* ── RIGHT SIDEBAR ── */}
        <aside className="w-72 shrink-0 flex flex-col gap-4 p-4 border-l border-border/50 bg-sidebar/60 backdrop-blur-md overflow-y-auto">

          <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-widest pt-1">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
            Live Session Monitor
          </div>

          {/* Phase + Turn */}
          <div className="glass-card rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Phase:</span>
                <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-primary/15 text-primary border border-primary/25">
                  {PHASES[phaseIndex]}
                </span>
                <span className="text-[11px] text-muted-foreground">({phaseTurnsDone}/{phaseTurnsTotal} turns)</span>
              </div>
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <span className="font-medium text-foreground">Turn:</span>
                <span className="font-semibold text-primary">{turn}</span>
              </div>
            </div>

            <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${(phaseIndex / (PHASES.length - 1)) * 100}%`,
                  background: "linear-gradient(90deg, hsl(33 95% 55%), hsl(175 60% 40%))",
                }}
              />
            </div>

            <p className="text-[11px] text-muted-foreground">
              Assessing {PHASES[phaseIndex].toLowerCase()} ({phaseTurnsDone}/{phaseTurnsTotal})
            </p>
          </div>

          {/* Domain Confidence */}
          <div className="glass-card rounded-xl p-4 space-y-3">
            <div className="text-xs font-semibold text-foreground">Domain Confidence</div>
            <div className="space-y-3">
              {DOMAINS.map(domain => {
                const val = confidence[domain.key] ?? 0;
                return (
                  <div key={domain.key} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] text-muted-foreground">{domain.label}</span>
                      <span className="text-[11px] font-medium text-foreground">{val}%</span>
                    </div>
                    <div className="w-full h-1.5 bg-muted/60 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-1000 ease-out"
                        style={{
                          width: `${val}%`,
                          background: val >= 70
                            ? "linear-gradient(90deg, hsl(175 60% 40%), hsl(175 60% 50%))"
                            : val >= 30
                            ? "linear-gradient(90deg, hsl(33 95% 55%), hsl(45 90% 60%))"
                            : "hsl(220 20% 30%)",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Progress tracker */}
          <div className="glass-card rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold text-foreground">Progress</div>
              <div className="text-[11px] text-muted-foreground">
                {domainsAssessed.length}/5 domains assessed
              </div>
            </div>

            <div className="flex items-center justify-center py-2">
              <div className="relative w-20 h-20">
                <svg className="w-full h-full -rotate-90" viewBox="0 0 80 80">
                  <circle cx="40" cy="40" r="30" fill="none" stroke="hsl(220 20% 20%)" strokeWidth="6"/>
                  <circle
                    cx="40" cy="40" r="30" fill="none"
                    stroke="url(#progress-grad)"
                    strokeWidth="6"
                    strokeLinecap="round"
                    strokeDasharray={`${2 * Math.PI * 30}`}
                    strokeDashoffset={`${2 * Math.PI * 30 * (1 - domainsAssessed.length / 5)}`}
                    className="transition-all duration-1000 ease-out"
                  />
                  <defs>
                    <linearGradient id="progress-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="hsl(33 95% 55%)"/>
                      <stop offset="100%" stopColor="hsl(175 60% 40%)"/>
                    </linearGradient>
                  </defs>
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-lg font-semibold text-foreground">{domainsAssessed.length}</span>
                  <span className="text-[9px] text-muted-foreground">of 5</span>
                </div>
              </div>
            </div>

            <div className="space-y-1.5">
              {DOMAINS.map(domain => {
                const done = domainsAssessed.includes(domain.key);
                return (
                  <div key={domain.key} className={`flex items-center gap-2 text-[11px] transition-colors ${done ? "text-foreground" : "text-muted-foreground/50"}`}>
                    <span className={`w-4 h-4 rounded-full flex items-center justify-center shrink-0 transition-all ${done ? "bg-secondary/30 text-secondary border border-secondary/40" : "border border-border/40"}`}>
                      {done && (
                        <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5"/></svg>
                      )}
                    </span>
                    {domain.label}
                  </div>
                );
              })}
            </div>

            {domainsAssessed.length < 5 && (
              <p className="text-[10px] text-muted-foreground/60 pt-1">
                Remaining: {DOMAINS.filter(d => !domainsAssessed.includes(d.key)).map(d => d.label).join(", ")}
              </p>
            )}
          </div>

        </aside>
      </div>
    </div>
  );
}
