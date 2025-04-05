---
title: Kokoro Text-to-Speech
emoji: 🗣️
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: apache-2.0
short_description: High-quality speech synthesis powered by Kokoro TTS
header: mini
models:
  - onnx-community/Kokoro-82M-ONNX
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# Kokoro Text-to-Speech

A simple React + Vite application for running [Kokoro](https://github.com/hexgrad/kokoro), a frontier text-to-speech model for its size. The model runs 100% locally in the browser using [kokoro-js](https://www.npmjs.com/package/kokoro-js) and [🤗 Transformers.js](https://www.npmjs.com/package/@huggingface/transformers)!

## Getting Started

Follow the steps below to set up and run the application.

### 1. Clone the Repository

```sh
git clone https://github.com/hexgrad/kokoro.git
```

### 2. Build the Dependencies

```sh
cd kokoro/kokoro.js
npm i
npm run build
```

### 3. Setup the Demo Project

Note this depends on build output from the previous step.

```sh
cd demo
npm i
```

### 4. Start the Development Server

```sh
npm run dev
```

The application should now be running locally. Open your browser and go to [http://localhost:5173](http://localhost:5173) to see it in action.
