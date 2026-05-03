import { useState, useEffect, useRef } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";

// Flores-200 → ISO 639-1 (what Streamlit uses)
const FLORES_TO_ISO: Record<string, string> = {
  eng_Latn: "en", hin_Deva: "hi", ben_Beng: "bn", tam_Taml: "ta",
  tel_Telu: "te", mal_Mlym: "ml", kan_Knda: "kn", mar_Deva: "mr",
  guj_Gujr: "gu", pan_Guru: "pa", urd_Arab: "ur", ory_Orya: "or",
  asm_Beng: "as", npi_Deva: "ne", mai_Deva: "hi", sat_Olck: "en",
  kas_Arab: "ur", kas_Deva: "hi", mni_Beng: "bn", mni_Mtei: "en",
  doi_Deva: "hi", brx_Deva: "hi", gom_Deva: "mr", san_Deva: "hi",
  snd_Arab: "ur", snd_Deva: "hi",
};

const LANGUAGES = [
  { code: "eng_Latn",  label: "English",              script: "Latin",       scriptGroup: "other",    native: "English",        rtl: false },
  { code: "hin_Deva",  label: "Hindi",                script: "Devanagari",  scriptGroup: "devanagari", native: "हिन्दी",        rtl: false },
  { code: "ben_Beng",  label: "Bengali",              script: "Bengali",     scriptGroup: "dravidian", native: "বাংলা",         rtl: false },
  { code: "tam_Taml",  label: "Tamil",                script: "Tamil",       scriptGroup: "dravidian", native: "தமிழ்",        rtl: false },
  { code: "tel_Telu",  label: "Telugu",               script: "Telugu",      scriptGroup: "dravidian", native: "తెలుగు",       rtl: false },
  { code: "mal_Mlym",  label: "Malayalam",            script: "Malayalam",   scriptGroup: "dravidian", native: "മലയാളം",       rtl: false },
  { code: "kan_Knda",  label: "Kannada",              script: "Kannada",     scriptGroup: "dravidian", native: "ಕನ್ನಡ",        rtl: false },
  { code: "mar_Deva",  label: "Marathi",              script: "Devanagari",  scriptGroup: "devanagari", native: "मराठी",        rtl: false },
  { code: "guj_Gujr",  label: "Gujarati",             script: "Gujarati",    scriptGroup: "other",    native: "ગુજરાતી",       rtl: false },
  { code: "pan_Guru",  label: "Punjabi",              script: "Gurmukhi",    scriptGroup: "other",    native: "ਪੰਜਾਬੀ",       rtl: false },
  { code: "urd_Arab",  label: "Urdu",                 script: "Perso-Arabic", scriptGroup: "persoarabic", native: "اردو",      rtl: true  },
  { code: "ory_Orya",  label: "Odia",                 script: "Odia",        scriptGroup: "other",    native: "ଓଡ଼ିଆ",         rtl: false },
  { code: "asm_Beng",  label: "Assamese",             script: "Bengali",     scriptGroup: "dravidian", native: "অসমীয়া",      rtl: false },
  { code: "mai_Deva",  label: "Maithili",             script: "Devanagari",  scriptGroup: "devanagari", native: "मैथिली",      rtl: false },
  { code: "sat_Olck",  label: "Santali",              script: "Ol Chiki",    scriptGroup: "other",    native: "ᱥᱟᱱᱛᱟᱲᱤ",    rtl: false },
  { code: "kas_Arab",  label: "Kashmiri (Arabic)",    script: "Perso-Arabic", scriptGroup: "persoarabic", native: "کٲشُر",    rtl: true  },
  { code: "kas_Deva",  label: "Kashmiri (Dev.)",      script: "Devanagari",  scriptGroup: "devanagari", native: "कॉशुर",      rtl: false },
  { code: "npi_Deva",  label: "Nepali",               script: "Devanagari",  scriptGroup: "devanagari", native: "नेपाली",      rtl: false },
  { code: "mni_Beng",  label: "Manipuri (Bengali)",   script: "Bengali",     scriptGroup: "dravidian", native: "মৈতেই",       rtl: false },
  { code: "mni_Mtei",  label: "Manipuri (Meitei)",    script: "Meitei",      scriptGroup: "other",    native: "ꯃꯩꯇꯩꯂꯣꯟ",    rtl: false },
  { code: "doi_Deva",  label: "Dogri",                script: "Devanagari",  scriptGroup: "devanagari", native: "डोगरी",      rtl: false },
  { code: "brx_Deva",  label: "Bodo",                 script: "Devanagari",  scriptGroup: "devanagari", native: "बड़ो",        rtl: false },
  { code: "gom_Deva",  label: "Konkani",              script: "Devanagari",  scriptGroup: "devanagari", native: "कोंकणी",     rtl: false },
  { code: "san_Deva",  label: "Sanskrit",             script: "Devanagari",  scriptGroup: "devanagari", native: "संस्कृतम्",  rtl: false },
  { code: "snd_Arab",  label: "Sindhi (Arabic)",      script: "Perso-Arabic", scriptGroup: "persoarabic", native: "سنڌي",    rtl: true  },
  { code: "snd_Deva",  label: "Sindhi (Dev.)",        script: "Devanagari",  scriptGroup: "devanagari", native: "सिन्धी",    rtl: false },
];

const TRANSLATIONS: Record<string, { headline: string; subtitle: string; moodHeading: string; }> = {
  "eng_Latn": { headline: "You Are Not Alone. We Are Here to Help.", subtitle: "Talk to our health assistant — anytime.", moodHeading: "How are you feeling right now?" },
  "hin_Deva": { headline: "आप अकेले नहीं हैं। हम यहाँ हैं।", subtitle: "हमारे स्वास्थ्य सहायक से बात करें — कभी भी।", moodHeading: "अभी आप कैसा महसूस कर रहे हैं?" },
  "ben_Beng": { headline: "আপনি একা নন। আমরা সাহায্য করতে এসেছি।", subtitle: "আমাদের স্বাস্থ্য সহকারীর সাথে কথা বলুন।", moodHeading: "আপনি এখন কেমন অনুভব করছেন?" },
  "tam_Taml": { headline: "நீங்கள் தனியல்ல. நாங்கள் உதவ இங்கே இருக்கிறோம்.", subtitle: "எங்கள் உடல்நல உதவியாளரிடம் பேசுங்கள்.", moodHeading: "நீங்கள் இப்போது எப்படி உணர்கிறீர்கள்?" },
  "tel_Telu": { headline: "మీరు ఒంటరిగా లేరు. మేము సహాయం చేయడానికి ఇక్కడ ఉన్నాం.", subtitle: "మా ఆరోగ్య సహాయకుడితో మాట్లాడండి.", moodHeading: "మీరు ఇప్పుడు ఎలా అనుభవిస్తున్నారు?" },
  "mal_Mlym": { headline: "നിങ്ങൾ ഒറ്റയ്ക്കല്ല. ഞങ്ങൾ സഹായിക്കാൻ ഇവിടെ ഉണ്ട്.", subtitle: "ഞങ്ങളുടെ ആരോഗ്യ സഹായിയോട് സംസാരിക്കൂ.", moodHeading: "നിങ്ങൾ ഇപ്പോൾ എങ്ങനെ അനുഭവിക്കുന്നു?" },
  "kan_Knda": { headline: "ನೀವು ಒಂಟಿಯಾಗಿಲ್ಲ. ನಾವು ಸಹಾಯ ಮಾಡಲು ಇಲ್ಲಿದ್ದೇವೆ.", subtitle: "ನಮ್ಮ ಆರೋಗ್ಯ ಸಹಾಯಕರೊಂದಿಗೆ ಮಾತನಾಡಿ.", moodHeading: "ನೀವು ಈಗ ಹೇಗೆ ಅನಿಸಿಕೊಳ್ಳುತ್ತಿದ್ದೀರಿ?" },
  "mar_Deva": { headline: "तुम्ही एकटे नाही. आम्ही मदतीसाठी इथे आहोत.", subtitle: "आमच्या आरोग्य सहाय्यकाशी बोला.", moodHeading: "तुम्हाला आत्ता कसे वाटते?" },
  "guj_Gujr": { headline: "તમે એકલા નથી. અમે મદદ કરવા અહીં છીએ.", subtitle: "અમારા આરોગ્ય સહાયક સાથે વાત કરો.", moodHeading: "તમે અત્યારે કેવું અનુભવ કરો છો?" },
  "pan_Guru": { headline: "ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ। ਅਸੀਂ ਤੁਹਾਡੀ ਮਦਦ ਲਈ ਇੱਥੇ ਹਾਂ।", subtitle: "ਸਾਡੇ ਸਿਹਤ ਸਹਾਇਕ ਨਾਲ ਗੱਲ ਕਰੋ।", moodHeading: "ਤੁਸੀਂ ਹੁਣ ਕਿਵੇਂ ਮਹਿਸੂਸ ਕਰਦੇ ਹੋ?" },
  "urd_Arab": { headline: "آپ اکیلے نہیں ہیں۔ ہم مدد کے لیے یہاں ہیں۔", subtitle: "ہمارے صحت معاون سے بات کریں۔", moodHeading: "آپ ابھی کیسا محسوس کر رہے ہیں؟" },
};

const MOODS = [
  { emoji: "😔", message: "You are safe here. Healing begins with one step." },
  { emoji: "😟", message: "We understand. Let us help you carry this." },
  { emoji: "😐", message: "That's okay. We're here whenever you're ready." },
  { emoji: "🙂", message: "Glad you're doing okay! We're here if you need us." },
  { emoji: "😊", message: "Wonderful! Share that energy — we're here for you." },
];

const SCRIPTS_STRIP = [
  { text: "We Are Here", name: "Latin" },
  { text: "हम यहाँ हैं", name: "Devanagari" },
  { text: "আমরা এখানে আছি", name: "Bengali" },
  { text: "நாங்கள் இங்கே இருக்கிறோம்", name: "Tamil" },
  { text: "ᱟᱢ ᱮᱛᱟᱦᱟᱸ ᱟᱫᱳᱜ", name: "Ol Chiki" }
];

export default function Home() {
  const [, navigate] = useLocation();
  const [activeLanguageCode, setActiveLanguageCode] = useState("eng_Latn");
  const [typedHeadline, setTypedHeadline] = useState("");
  const [showSubtitle, setShowSubtitle] = useState(false);
  const [scriptFilter, setScriptFilter] = useState("All");
  const [selectedMoodIndex, setSelectedMoodIndex] = useState<number | null>(null);
  const [scriptCarouselIndex, setScriptCarouselIndex] = useState(0);

  function goToChat(floresCode: string, moodIndex: number | null) {
    const iso = FLORES_TO_ISO[floresCode] ?? "en";
    const mood = moodIndex !== null ? String(moodIndex) : "";
    navigate(`/chat?lang=${iso}&mood=${mood}`);
  }

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cardsRef = useRef<HTMLDivElement>(null);

  const activeLang = LANGUAGES.find(l => l.code === activeLanguageCode) || LANGUAGES[0];
  const translation = TRANSLATIONS[activeLanguageCode] || {
    headline: TRANSLATIONS["eng_Latn"].headline,
    subtitle: "🔄 Live translation via IndicTrans2",
    moodHeading: TRANSLATIONS["eng_Latn"].moodHeading,
  };

  // Particles
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let particles: {x: number, y: number, vx: number, vy: number, size: number}[] = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", resize);
    resize();

    for (let i = 0; i < 40; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        size: Math.random() * 2 + 1,
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "rgba(220, 25%, 8%, 1)";
      
      particles.forEach(p => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(245, 158, 11, 0.4)"; // Saffron tint
        ctx.fill();
      });

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 150) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(20, 184, 166, ${1 - dist/150})`; // Teal tint
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      animationFrameId = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  // Typewriter
  useEffect(() => {
    let index = 0;
    setTypedHeadline("");
    setShowSubtitle(false);
    const fullText = translation.headline;
    
    const intervalId = setInterval(() => {
      setTypedHeadline(fullText.substring(0, index + 1));
      index++;
      if (index === fullText.length) {
        clearInterval(intervalId);
        setTimeout(() => setShowSubtitle(true), 300);
      }
    }, 50);

    return () => clearInterval(intervalId);
  }, [activeLanguageCode, translation.headline]);

  // Card staggered animation
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("opacity-100", "translate-y-0");
          entry.target.classList.remove("opacity-0", "translate-y-8");
        }
      });
    }, { threshold: 0.1 });

    if (cardsRef.current) {
      const cards = cardsRef.current.querySelectorAll('.glass-card-container');
      cards.forEach((card, idx) => {
        (card as HTMLElement).style.transitionDelay = `${idx * 150}ms`;
        observer.observe(card);
      });
    }

    return () => observer.disconnect();
  }, []);

  // Script carousel
  useEffect(() => {
    const id = setInterval(() => {
      setScriptCarouselIndex(prev => (prev + 1) % SCRIPTS_STRIP.length);
    }, 3000);
    return () => clearInterval(id);
  }, []);

  const handleRipple = (e: React.MouseEvent<HTMLButtonElement>) => {
    const button = e.currentTarget;
    const rect = button.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = e.clientX - rect.left - size / 2;
    const y = e.clientY - rect.top - size / 2;

    const span = document.createElement("span");
    span.style.width = span.style.height = `${size}px`;
    span.style.left = `${x}px`;
    span.style.top = `${y}px`;
    span.classList.add("ripple-span");

    button.appendChild(span);
    setTimeout(() => span.remove(), 600);
  };

  const filteredLanguages = scriptFilter === "All" 
    ? LANGUAGES 
    : LANGUAGES.filter(l => l.scriptGroup.toLowerCase() === scriptFilter.toLowerCase().replace("-", ""));

  return (
    <div className="min-h-screen bg-background text-foreground font-sans overflow-x-hidden selection:bg-primary/30 selection:text-primary">
      
      {/* Hero Section */}
      <section className="relative min-h-[100dvh] flex flex-col items-center justify-center pt-20 pb-10 px-4">
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none opacity-50" />
        
        <div className="relative z-10 w-full max-w-4xl mx-auto flex flex-col items-center text-center space-y-8">
          <div className="text-5xl md:text-7xl mb-4 animate-heartbeat">🪷</div>
          
          <div dir={activeLang.rtl ? "rtl" : "ltr"} className="min-h-[120px] md:min-h-[160px] flex flex-col items-center justify-center space-y-4">
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-serif font-medium tracking-tight text-foreground leading-tight drop-shadow-sm">
              {typedHeadline}
              <span className="animate-pulse text-primary ml-1">|</span>
            </h1>
            
            <p className={`text-xl md:text-2xl text-muted-foreground font-light max-w-2xl transition-all duration-1000 transform ${showSubtitle ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}>
              {translation.subtitle}
            </p>
          </div>

          <div className="pt-8">
            <Button
              size="lg"
              className="relative overflow-hidden bg-primary hover:bg-primary/90 text-primary-foreground rounded-full px-8 py-6 text-lg font-medium shadow-lg hover:shadow-primary/25 transition-all"
              onClick={(e) => { handleRipple(e); goToChat(activeLanguageCode, selectedMoodIndex); }}
            >
              Start Talking to Us →
            </Button>
          </div>

          <div className="inline-flex items-center gap-2 mt-8 px-4 py-2 rounded-full border border-primary/20 bg-background/50 backdrop-blur-sm text-sm text-muted-foreground">
            <span className="text-lg">🇮🇳</span> Powered by IndicTrans2 · 22 Indian Languages + English
          </div>
        </div>
      </section>

      {/* Language Selector */}
      <section className="py-20 px-4 bg-gradient-to-b from-background to-card/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-10">
            <h2 className="text-2xl font-serif mb-2 text-foreground">Find Your Language</h2>
            <p className="text-muted-foreground">Select your preferred language to personalize your experience.</p>
          </div>
          
          <div className="flex flex-wrap justify-center gap-2 mb-10">
            {["All", "Devanagari", "Dravidian", "Perso-Arabic", "Other Scripts"].map(filter => (
              <button
                key={filter}
                onClick={() => setScriptFilter(filter)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${scriptFilter === filter ? "bg-primary text-primary-foreground shadow-md" : "bg-card border border-border text-muted-foreground hover:text-foreground"}`}
              >
                {filter}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {filteredLanguages.map(lang => (
              <button
                key={lang.code}
                onClick={() => setActiveLanguageCode(lang.code)}
                className={`flex flex-col items-center justify-center p-4 rounded-xl transition-all duration-300 border ${activeLanguageCode === lang.code ? "border-primary bg-primary/10 shadow-[0_0_15px_rgba(245,158,11,0.15)] scale-105" : "border-border bg-card hover:border-primary/50 hover:bg-card/80"}`}
              >
                <span className="text-2xl font-serif mb-1">{lang.native}</span>
                <span className="text-xs text-muted-foreground">{lang.label}</span>
              </button>
            ))}
          </div>

          <div className="mt-8 flex justify-center h-8">
            <div className={`transition-all duration-500 ${activeLang ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}`}>
              <span className="px-3 py-1 rounded-full text-xs border border-secondary text-secondary bg-secondary/10">
                Script: {activeLang?.script}
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Feature Cards */}
      <section className="py-24 px-4 relative">
        <div className="max-w-5xl mx-auto" ref={cardsRef}>
          <div className="grid md:grid-cols-3 gap-6">
            {[
              { icon: "🌿", title: "Safe & Confidential", desc: "Everything you share stays between you and us." },
              { icon: "💬", title: "Always Listening", desc: "Our AI health assistant is here 24/7, no waiting." },
              { icon: "🇮🇳", title: "Your Language, Your Comfort", desc: "22 Indian languages, powered by IndicTrans2." }
            ].map((card, i) => (
              <div key={i} className="glass-card-container opacity-0 translate-y-8 transition-all duration-700 ease-out">
                <div className="glass-card rounded-2xl p-8 h-full flex flex-col items-center text-center hover:bg-white/5 transition-colors">
                  <div className="text-4xl mb-4">{card.icon}</div>
                  <h3 className="text-xl font-serif mb-3 text-foreground">{card.title}</h3>
                  <p className="text-muted-foreground leading-relaxed">{card.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Mood Check */}
      <section className="py-24 px-4 bg-gradient-to-b from-card/30 to-background relative overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-primary/5 rounded-full blur-[100px] pointer-events-none"></div>
        <div className="max-w-3xl mx-auto text-center relative z-10">
          <h2 className="text-3xl md:text-4xl font-serif mb-12 text-foreground transition-opacity" dir={activeLang.rtl ? "rtl" : "ltr"}>
            {translation.moodHeading}
          </h2>
          
          <div className="flex flex-wrap justify-center gap-4 md:gap-8 mb-12">
            {MOODS.map((mood, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedMoodIndex(idx)}
                className={`text-5xl md:text-6xl transition-all duration-300 hover:scale-125 hover:-translate-y-2 ${selectedMoodIndex === idx ? "scale-125 -translate-y-2 drop-shadow-[0_0_15px_rgba(245,158,11,0.5)] grayscale-0" : selectedMoodIndex !== null ? "opacity-40 grayscale" : ""}`}
              >
                {mood.emoji}
              </button>
            ))}
          </div>

          <div className={`transition-all duration-500 h-24 ${selectedMoodIndex !== null ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}>
            {selectedMoodIndex !== null && (
              <div className="flex flex-col items-center space-y-6">
                <p className="text-xl md:text-2xl text-primary font-medium">
                  {MOODS[selectedMoodIndex].message}
                </p>
                <Button
                  variant="outline"
                  className="rounded-full border-primary/50 hover:bg-primary hover:text-primary-foreground"
                  onClick={() => goToChat(activeLanguageCode, selectedMoodIndex)}
                >
                  Continue to Chat →
                </Button>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Script Showcase */}
      <section className="py-16 border-y border-border/50 bg-card/30">
        <div className="max-w-4xl mx-auto text-center px-4">
          <div className="h-20 flex items-center justify-center mb-4">
            {SCRIPTS_STRIP.map((script, idx) => (
              <h3 
                key={idx}
                className={`absolute text-4xl md:text-5xl font-serif text-secondary transition-opacity duration-1000 ${idx === scriptCarouselIndex ? "opacity-100" : "opacity-0"}`}
              >
                {script.text}
              </h3>
            ))}
          </div>
          <p className="text-muted-foreground text-sm">
            Showing <span className="text-foreground font-medium">{SCRIPTS_STRIP[scriptCarouselIndex].name}</span> · Script used by IndicTrans2
          </p>
          <div className="flex justify-center gap-2 mt-4 text-xs text-muted-foreground/60">
            Devanagari · Bengali · Tamil · Perso-Arabic · Ol Chiki · Meitei
          </div>
        </div>
      </section>

      {/* Testimonial Ticker */}
      <section className="py-12 overflow-hidden relative border-b border-border/50">
        <div className="absolute left-0 top-0 bottom-0 w-24 md:w-48 bg-gradient-to-r from-background to-transparent z-10 pointer-events-none"></div>
        <div className="absolute right-0 top-0 bottom-0 w-24 md:w-48 bg-gradient-to-l from-background to-transparent z-10 pointer-events-none"></div>
        
        <div className="flex w-[200%] animate-ticker opacity-70">
          <div className="flex w-1/2 justify-around items-center space-x-16 px-8">
            <span className="text-xl whitespace-nowrap font-serif text-muted-foreground">"इस सेवा ने मुझे बहुत मदद की।"</span>
            <span className="text-xl whitespace-nowrap font-serif text-muted-foreground">"Finally, help in my language."</span>
            <span className="text-xl whitespace-nowrap font-serif text-muted-foreground">"এই সেবা আমাকে সাহায্য করেছে।"</span>
            <span className="text-xl whitespace-nowrap font-serif text-muted-foreground">"இந்த சேவை மிகவும் உதவியது."</span>
          </div>
          <div className="flex w-1/2 justify-around items-center space-x-16 px-8">
            <span className="text-xl whitespace-nowrap font-serif text-muted-foreground">"ഈ സേവനം എന്നെ ഒരുപാട് സഹായിച്ചു."</span>
            <span className="text-xl whitespace-nowrap font-serif text-muted-foreground">"ਇਸ ਸੇਵਾ ਨੇ ਮੇਰੀ ਬਹੁਤ ਮਦਦ ਕੀਤੀ।"</span>
            <span className="text-xl whitespace-nowrap font-serif text-muted-foreground">"इस सेवा ने मुझे राह दिखाई।"</span>
            <span className="text-xl whitespace-nowrap font-serif text-muted-foreground">"ناز ہے اس سروس پر۔"</span>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-24 px-4 text-center bg-gradient-to-t from-primary/5 to-background relative">
        <div className="max-w-2xl mx-auto flex flex-col items-center space-y-8">
          <h2 className="text-3xl font-serif" dir={activeLang.rtl ? "rtl" : "ltr"}>{translation.headline}</h2>
          <p className="text-muted-foreground">No sign-up needed. Free. Private. Yours.</p>
          
          <Button
            size="lg"
            className="rounded-full px-10 py-8 text-xl group overflow-hidden relative bg-card border border-primary/30 hover:border-primary hover:bg-card text-foreground"
            onClick={() => goToChat(activeLanguageCode, selectedMoodIndex)}
          >
            <span className="relative z-10 flex items-center gap-2">
              Talk to Us <span className="group-hover:translate-x-1 transition-transform">💬</span>
            </span>
            <div className="absolute inset-0 bg-gradient-to-r from-primary/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></div>
          </Button>

          <p className="text-sm text-muted-foreground mt-12 opacity-50">
            Translation powered by AI4Bharat IndicTrans2
          </p>
        </div>
      </footer>

      {/* Floating Action Button */}
      <button
        className="fixed bottom-6 right-6 md:bottom-10 md:right-10 bg-primary text-primary-foreground p-4 rounded-full shadow-lg pulse-ring hover:scale-110 transition-transform z-50 flex items-center justify-center font-medium gap-2"
        onClick={() => goToChat(activeLanguageCode, selectedMoodIndex)}
      >
        <span className="text-xl">💬</span> <span className="hidden md:inline">Need help now?</span>
      </button>

    </div>
  );
}
