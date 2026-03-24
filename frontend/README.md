# Iikshana Frontend

React Progressive Web App designed for blind courtroom participants. The interface prioritizes screen reader compatibility, keyboard navigation, and high contrast visuals, targeting WCAG 2.1 Level AAA compliance.

## Folder Structure

```text
frontend/
├── src/
│   ├── components/
│   │   ├── AccessibilityControls.tsx  # Playback speed, volume, emotion toggle (WCAG AAA)
│   │   ├── AudioCapture.tsx           # Microphone access via Web Audio API, streams to backend
│   │   ├── ControlPanel.tsx           # Keyboard accessible preferences and session controls
│   │   ├── TranscriptDisplay.tsx      # Speaker labeled transcript with ARIA structure
│   │   └── ImageViewer.tsx            # Visual evidence viewer with accessible captions
│   ├── hooks/
│   │   ├── useWebSocket.ts            # WebSocket connection state and event handling
│   │   ├── useAudioCapture.ts         # Microphone access and audio streaming
│   │   └── useKeyboardShortcuts.ts    # Keyboard shortcut listeners (S, P, R, E, I, H)
│   ├── services/
│   │   ├── api.ts                     # HTTP client (Axios) for REST endpoints
│   │   └── audio.ts                   # Audio playback via Web Audio API, queue management
│   ├── types/                         # TypeScript type definitions
│   ├── utils/                         # Shared helper functions
│   ├── App.tsx                        # Root component, state management, routing
│   └── index.tsx                      # React entry point
├── public/                            # Static assets and PWA manifest
├── package.json
└── tsconfig.json
```

## Components

**AccessibilityControls** provides settings for playback speed, volume, and emotion toggle. All controls are built to meet WCAG 2.1 Level AAA standards with proper ARIA labels and focus management.

**AudioCapture** uses the Web Audio API to access the user's microphone and streams audio chunks to the backend over WebSocket.

**ControlPanel** is the main control surface for session preferences and system actions. It implements keyboard shortcuts (S, P, R, E, I, H) so users can operate the interface without a mouse.

**TranscriptDisplay** renders the live courtroom transcript with speaker labels and color coding. The structure is optimized for screen readers using ARIA roles and live regions so new transcript segments are announced automatically.

**ImageViewer** handles visual evidence (photos, documents). When an image is loaded, it requests a caption from the backend's Vision agent and displays it in an accessible format.

## Custom Hooks

| Hook | Purpose |
|------|---------|
| `useWebSocket` | Manages the WebSocket connection lifecycle, reconnection, and message dispatching |
| `useAudioCapture` | Handles microphone permissions, audio context setup, and chunk streaming |
| `useKeyboardShortcuts` | Registers global keyboard listeners for the shortcut keys |

## Services

**api.ts** wraps Axios for HTTP calls to the backend (session management, image uploads, configuration endpoints).

**audio.ts** manages TTS audio playback through the Web Audio API. It queues incoming audio segments and handles playback controls (pause, resume, skip).

## Getting Started

### Prerequisites

- Node.js 18 or higher
- npm (comes with Node)

### Install and Run

```bash
cd frontend
npm install
npm start
```

The app runs at `http://localhost:3000` by default, using React Scripts (Create React App).

### Build for Production

```bash
npm run build
```

This creates an optimized production build in `frontend/build/`.

### Run Tests

```bash
npm test
```

## TypeScript Configuration

The project uses strict TypeScript (`strict: true`) with ES2020 target and React JSX transform. The full config is in `tsconfig.json`. Module resolution follows the Node strategy.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| S | Start/stop audio capture |
| P | Pause/resume playback |
| R | Replay last segment |
| E | Toggle emotion indicators |
| I | Request image description |
| H | Open help/shortcuts panel |

These shortcuts are registered globally through the `useKeyboardShortcuts` hook and work regardless of which component has focus.

## Connecting to the Backend

The frontend connects to the backend via two channels:

1. **WebSocket** (socket.io) for real time audio streaming and transcript updates
2. **HTTP** (Axios) for session management, image uploads, and configuration

By default the frontend expects the backend at `http://localhost:8000`. This can be configured through environment variables or the API service module.
