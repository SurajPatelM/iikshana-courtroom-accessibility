import { useState } from "react";
import HeroBanner from "./components/HeroBanner";
import BatchProcessing from "./components/BatchProcessing";
import RealtimeProcessing from "./components/RealtimeProcessing";
import "./App.css";

type Tab = "batch" | "realtime";

export default function App() {
  const [tab, setTab] = useState<Tab>("batch");

  return (
    <div className="app">
      <HeroBanner />

      <div className="tabs">
        <button
          className={`tab ${tab === "batch" ? "active" : ""}`}
          onClick={() => setTab("batch")}
        >
          🔄 Batch Processing
        </button>
        <button
          className={`tab ${tab === "realtime" ? "active" : ""}`}
          onClick={() => setTab("realtime")}
        >
          ⚡ Real-Time Processing
        </button>
      </div>

      <div className="tab-content">
        {tab === "batch" ? <BatchProcessing /> : <RealtimeProcessing />}
      </div>

      <div className="footer">
        <p>
          ⚖️ IIKSHANA COURTROOM ACCESSIBILITY &nbsp;·&nbsp; Empowering equal
          access to justice through audio AI
        </p>
      </div>
    </div>
  );
}