import { useRef, useState, useEffect } from "react";

export default function App() {
  // Create a reference to the worker object.
  const worker = useRef(null);
  const [inputText, setInputText] = useState(
    "Life is like a box of chocolates. You never know what you're gonna get.",
  );
  const [selectedSpeaker, setSelectedSpeaker] = useState("af_heart");
  const [voices, setVoices] = useState([]);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [loadingMessage, setLoadingMessage] = useState("Loading...");
  const [results, setResults] = useState([]);

  // We use the `useEffect` hook to setup the worker as soon as the `App` component is mounted.
  useEffect(() => {
    // Create the worker if it does not yet exist.
    worker.current ??= new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
    });

    // Create a callback function for messages from the worker thread.
    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case "device":
          setLoadingMessage(`Loading model (device="${e.data.device}")`);
          break;
        case "ready":
          setStatus("ready");
          setVoices(e.data.voices);
          break;
        case "error":
          setError(e.data.data);
          break;
        case "complete": {
          const { audio, text } = e.data;
          // Generation complete: re-enable the "Generate" button
          setResults((prev) => [{ text, src: audio }, ...prev]);
          setStatus("ready");
        }
      }
    };

    const onErrorReceived = (e) => {
      console.error("Worker error:", e);
      setError(e.message);
    };

    // Attach the callback function as an event listener.
    worker.current.addEventListener("message", onMessageReceived);
    worker.current.addEventListener("error", onErrorReceived);

    // Define a cleanup function for when the component is unmounted.
    return () => {
      worker.current.removeEventListener("message", onMessageReceived);
      worker.current.removeEventListener("error", onErrorReceived);
    };
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    setStatus("running");

    worker.current.postMessage({
      type: "generate",
      text: inputText.trim(),
      voice: selectedSpeaker,
    });
  };

  return (
    <div className="relative w-full min-h-screen bg-linear-to-br from-gray-900 to-gray-700 flex flex-col items-center justify-center p-4 overflow-hidden font-sans">
      <div
        className="absolute w-screen h-screen transition-opacity duration-500 justify-center flex flex-col items-center z-10 bg-gray-800/95 backdrop-blur-md"
        style={{
          opacity: status === null ? 1 : 0,
          pointerEvents: status === null ? "auto" : "none",
        }}
      >
        <div className="size-62.5 border-4 border-white shadow-[0_0_0_5px_#4973ff] rounded-full overflow-hidden">
          <div className="loading-wave"></div>
        </div>
        <p className={`text-3xl my-5 text-center ${error ? "text-red-500" : "text-white"}`}>
          {error ?? loadingMessage}
        </p>
      </div>

      <div className="max-w-3xl w-full space-y-8 relative z-2">
        <div className="text-center">
          <h1 className="text-5xl font-extrabold text-gray-100 mb-2 drop-shadow-lg font-heading">
            Kokoro Text-to-Speech
          </h1>
          <p className="text-2xl text-gray-300 font-semibold font-subheading">
            Powered by&nbsp;
            <a
              href="https://github.com/hexgrad/kokoro"
              target="_blank"
              rel="noreferrer"
              className="underline"
            >
              Kokoro
            </a>
            &nbsp;and&nbsp;
            <a
              href="https://huggingface.co/docs/transformers.js"
              target="_blank"
              rel="noreferrer"
              className="underline"
            >
              <img width="40" src="hf-logo.svg" className="inline -translate-y-0.5 me-1"></img>
              Transformers.js
            </a>
          </p>
        </div>
        <div className="bg-gray-800/50 backdrop-blur-xs border border-gray-700 rounded-lg p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <textarea
              placeholder="Enter text..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="w-full min-h-25 max-h-75 bg-gray-700/50 backdrop-blur-xs border-2 border-gray-600 rounded-xl resize-y text-gray-100 placeholder:text-gray-400 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={Math.min(8, inputText.split("\n").length)}
            />
            <div className="flex flex-col items-center space-y-4">
              <select
                value={selectedSpeaker}
                onChange={(e) => setSelectedSpeaker(e.target.value)}
                className="w-full bg-gray-700/50 backdrop-blur-xs border-2 border-gray-600 rounded-xl text-gray-100 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {Object.entries(voices).map(([id, voice]) => (
                  <option key={id} value={id}>
                    {voice.name} ({voice.language === "en-us" ? "American" : "British"}{" "}
                    {voice.gender})
                  </option>
                ))}
              </select>
              <button
                type="submit"
                className="inline-flex justify-center items-center px-6 py-2 text-lg font-semibold bg-linear-to-t from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 transition-colors duration-300 rounded-xl text-white disabled:opacity-50"
                disabled={status === "running" || inputText.trim() === ""}
              >
                {status === "running" ? "Generating..." : "Generate"}
              </button>
            </div>
          </form>
        </div>

        {results.length > 0 && (
          <div className="starting:opacity-0 starting:translate-y-12.5 transition-[translate,opacity] duration-500 opacity-100 translate-y-0 max-h-62.5 overflow-y-auto px-2 mt-4 space-y-6 relative z-2">
            {results.map((result, i) => (
              <div key={i}>
                <div className="text-white bg-gray-800/70 backdrop-blur-xs border border-gray-700 rounded-lg p-4 z-10">
                  <span className="absolute right-5 font-bold">#{results.length - i}</span>
                  <p className="mb-3 max-w-[95%]">{result.text}</p>
                  <audio controls src={result.src} className="w-full">
                    Your browser does not support the audio element.
                  </audio>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="bg-[#015871] pointer-events-none absolute left-0 w-full h-[5%] -bottom-12.5">
        <div className="wave"></div>
        <div className="wave"></div>
      </div>
    </div>
  );
}
