# AarogyaVaani — Multilingual Health Chatbot Landing Page

A multilingual health chatbot landing page powered by IndicTrans2, supporting all 22 scheduled Indian languages + English.

## Prerequisites

- [Node.js](https://nodejs.org/) v18 or later
- npm (comes with Node.js)

## Setup

```bash
# Install dependencies
npm install

# Start the development server
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000) in your browser.

## Build for production

```bash
npm run build
npm run preview
```

## Tech stack

- React 19
- TypeScript
- Vite
- Tailwind CSS v4
- shadcn/ui components
- Framer Motion
- Wouter (routing)

## Features

- Typewriter hero headline in the active language
- 26-language selector with script-family grouping tabs
- Automatic RTL layout for Urdu, Kashmiri (Arabic), Sindhi (Arabic)
- Canvas-based animated particle mesh background
- Interactive mood check with personalised responses
- Script showcase carousel (Devanagari, Bengali, Tamil, Perso-Arabic, Ol Chiki, Meitei)
- Infinite testimonial ticker in mixed Indian scripts
- Floating "Need help now?" button
- Glassmorphism feature cards with staggered entrance animations
- Fully mobile responsive (375px+)
